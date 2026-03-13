"""
Utility functions for the ELiC LiDAR geometry compression framework.

These routines provide Morton coding and Morton-order-based hierarchical voxel
processing used in the ELiC method. Instead of pointer-driven octree traversal
or conventional sparse-tensor 2x2x2 down-convolutions with 1/2/4/8-style
occupancy labeling, they construct occupancy symbols directly from integer
coordinates and Morton codes using grouping and bit operations. This yields a
vectorized, GPU-friendly implementation that reduces reliance on sparse-conv
kernels and makes hierarchical geometry processing simpler and more efficient.
"""
import torch
from typing import Tuple

#####################################################################################
@torch.no_grad()
def _dilate3_u64(v: torch.Tensor) -> torch.Tensor:
    """
    Bit-dilates a 21-bit integer tensor for 3D Morton encoding.

    Each input value v (assumed to be in [0, 2^21)) is converted to a 64-bit
    integer where the original bits of v are spread out so that there are
    two zero bits between consecutive bits of v. In other words, the i-th bit
    of v is moved to bit position 3*i in the output.

    This "bit spreading" (dilation) is the standard pre-processing step used
    to build a 3D Morton code (Z-order curve), where the final Morton index
    is obtained by interleaving the dilated x, y, z coordinates.
    """
    v = v.to(torch.int64) & 0x1fffff
    v = (v | (v << 32)) & 0x1f00000000ffff
    v = (v | (v << 16)) & 0x1f0000ff0000ff
    v = (v | (v << 8))  & 0x100f00f00f00f00f
    v = (v | (v << 4))  & 0x10c30c30c30c30c3
    v = (v | (v << 2))  & 0x1249249249249249
    return v

#####################################################################################
@torch.no_grad()
def morton3_code(xyz: torch.Tensor) -> torch.Tensor:
    """
    Computes 3D Morton (Z-order) codes from integer 3D coordinates.

    Args:
        xyz: (N, 3) integer tensor of 3D coordinates [x, y, z],
             typically with each component limited to 21 bits (0 <= x,y,z < 2^21).

    Returns:
        (N,) 64-bit integer tensor where each element is the 3D Morton code
        obtained by bit-interleaving the bits of x, y, and z:
            morton = interleave_bits(x, y, z)

    Implementation details:
        - _dilate3_u64(x) spreads the bits of x so that there are two zero bits
          between consecutive bits (i.e., bits go to positions 0, 3, 6, ...).
        - _dilate3_u64(y) is shifted left by 1 so its bits occupy positions
          1, 4, 7, ...
        - _dilate3_u64(z) is shifted left by 2 so its bits occupy positions
          2, 5, 8, ...
        - The final Morton code is the bitwise OR of these three dilated values.
    """
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    return _dilate3_u64(x) | (_dilate3_u64(y) << 1) | (_dilate3_u64(z) << 2)

#####################################################################################
@torch.no_grad()
def morton3_sort(bxyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sorts 3D voxel coordinates in Morton (Z-order) order.

    Args:
        bxyz: (N, 4) integer tensor where each row is [batch_id, x, y, z].
              Duplicate rows are removed before sorting.

    Returns:
        bxyz_sorted: (M, 4) tensor of unique coordinates sorted by their Morton code.
        code_sorted: (M,) 64-bit Morton codes corresponding to bxyz_sorted.

    Procedure:
        1. Remove duplicate voxel coordinates via torch.unique(..., dim=0).
        2. Compute the Morton code of each (x, y, z) triple using morton3_code().
        3. Sort all voxels by their Morton code in ascending Z-order.
        4. Return the sorted coordinates and their associated codes.

    The output ordering follows the 3D Z-order curve, which preserves spatial locality.
    """
    bxyz = torch.unique(bxyz, dim=0)
    code = morton3_code(bxyz[:,1:])
    idx = torch.argsort(code)
    bxyz = bxyz[idx]
    code = code[idx]
    return bxyz, code

#####################################################################################
@torch.no_grad()
def down_once(bxyz: torch.Tensor, code: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregates one octree level down: from child voxels to their parent voxels.

    Given Morton-sorted leaf voxels, this function groups them by parent cell
    (coarser level) and computes:

        - parent_bxyz : (N_parent, 4) int32
            Unique parent voxel coordinates [batch_id, x, y, z].
            Here, batch_id is set to 0, and x,y,z are the integer parent coordinates
            obtained by floor-dividing the child coordinates by 2.

        - parent_code : (N_parent,) same dtype as input `code`
            Morton code of each parent voxel, obtained by shifting the child
            Morton codes right by 3 bits (removing one octree level).

        - occ_sym : (N_parent,) int32
            Per-parent 8-bit occupancy symbol (stored in an int32) that encodes
            which of the 8 children are present:
                bit 0: child (0,0,0)
                bit 1: child (1,0,0)
                bit 2: child (0,1,0)
                bit 3: child (1,1,0)
                bit 4: child (0,0,1)
                bit 5: child (1,0,1)
                bit 6: child (0,1,1)
                bit 7: child (1,1,1)

    Requirements:
        - `bxyz` is (N, 4) with integer coordinates [b, x, y, z].
        - `code` is (N,) Morton codes corresponding to those coordinates.
        - `code` and `bxyz` are assumed to be sorted by Morton code so that
          identical parent_code values appear consecutively.

    Steps:
        1. Compute parent coordinates and parent Morton codes by shifting
           child xyz and code:
                parent_xyz  = xyz >> 1
                parent_code = code >> 3
        2. Use torch.unique_consecutive on parent_code to:
                - get unique parent_code
                - 'inv' mapping from each child to its parent index
                - 'cnt' number of children per parent
        3. For each child, compute its local octant index (0..7) from the LSB
           of (x, y, z), and convert it to a one-hot bit (1 << pos).
        4. Accumulate these bits over children with the same parent (scatter_add),
           producing a compact occupancy symbol per parent.
        5. For each parent, take the coordinates of the first child in its group
           (according to 'starts') and use parent_xyz from that child as the
           unique parent coordinate.

    Returns:
        parent_bxyz, parent_code, occ_sym
    """
    device = bxyz.device
    xyz = bxyz[:, 1:]
    
    parent_xyz  = (xyz >> 1) # child coordinates // 2
    parent_code = (code >> 3) # remove one octree level (3 bits) from the Morton code
    parent_code, inv, cnt = torch.unique_consecutive(
        parent_code, return_inverse=True, return_counts=True)
    
    starts = torch.zeros_like(cnt)
    starts[1:] = torch.cumsum(cnt, dim=0)[:-1]
    first_idx = starts
    
    # Extract the least significant bits of (x, y, z) and encode them as a 3-bit octant index (0–7)
    pos = (xyz[:, 0] & 1) + ((xyz[:, 1] & 1) << 1) + ((xyz[:, 2] & 1) << 2)
    # One-hot occupancy bit for each child voxel (1 << pos in [1, 128])
    occ_bits = (1 << pos).to(torch.int32)      
    N = parent_code.numel() # Number of parent voxels
    
    # Sum occupancy bits over children that share the same parent_code
    occ_sym = torch.zeros(N, dtype=torch.int32, device=device) 
    occ_sym.scatter_add_(0, inv, occ_bits)
    
    parent_bxyz = torch.empty((N, 4), dtype=torch.int32, device=device)
    parent_bxyz[:, 0] = 0
    parent_bxyz[:, 1:] = parent_xyz[first_idx].to(torch.int32) # unique parent coordinates
    
    return parent_bxyz, parent_code, occ_sym

#####################################################################################
@torch.no_grad()
def _get_occ_bits(device=None) -> torch.Tensor:
    """
    Returns a static (256, 8) bool tensor
    occ_bits[o, c] = 1 if child c (0..7) is occupied in occupancy symbol o (0..255).
    """
    if not hasattr(_get_occ_bits, "_occ_bits"):
        d = torch.device("cuda") if device is None else device
        occ = torch.arange(256, dtype=torch.int32, device=d)[:, None]  # (256, 1)
        pos = torch.arange(8,   dtype=torch.int32, device=d)[None, :]  # (1, 8)
        _get_occ_bits._occ_bits = ((occ >> pos) & 1).to(torch.bool)    # (256, 8)

    if device is not None and _get_occ_bits._occ_bits.device != device:
        _get_occ_bits._occ_bits = _get_occ_bits._occ_bits.to(device)
    return _get_occ_bits._occ_bits

#####################################################################################
@torch.no_grad()
def _get_expand_coords_base(device=None) -> torch.Tensor:
    """
    Returns a static (8, 3) tensor for child coordinate offsets:
        [0,0,0], [1,0,0], [0,1,0], [1,1,0],
        [0,0,1], [1,0,1], [0,1,1], [1,1,1]
    """
    if not hasattr(_get_expand_coords_base, "_table"):
        d = torch.device("cuda") if device is None else device
        _get_expand_coords_base._table = torch.tensor([
            [0, 0, 0],  # -> 1
            [1, 0, 0],  # -> 2
            [0, 1, 0],  # -> 4
            [1, 1, 0],  # -> 8
            [0, 0, 1],  # -> 16
            [1, 0, 1],  # -> 32
            [0, 1, 1],  # -> 64
            [1, 1, 1],  # -> 128
        ], device=d, dtype=torch.int32)

    if device is not None and _get_expand_coords_base._table.device != device:
        _get_expand_coords_base._table = _get_expand_coords_base._table.to(device)
    return _get_expand_coords_base._table


#####################################################################################
def upscale_feature(x_O: torch.Tensor, x_F: torch.Tensor) -> torch.Tensor:
    """
    Expands per-parent features to their occupied child voxels using an
    8-way occupancy symbol.

    Args:
        x_O: (N,) int tensor of occupancy symbols in [0, 255].
             Each symbol encodes which of the 8 children are present:
                 bit 0: child (0,0,0)
                 bit 1: child (1,0,0)
                 ...
                 bit 7: child (1,1,1)
        x_F: (N, C) tensor of features at the parent (coarse) level.

    Returns:
        x_up_F: (N_up, C) tensor of features at the child (fine) level,
                containing one feature vector for each occupied child voxel.

    Implementation details:
        - _get_occ_bits() returns a (256, 8) boolean table mapping each
          occupancy symbol (0..255) to which of the 8 child slots are occupied.
        - For each parent i, x_F[i] is conceptually replicated to all 8 children.
        - The boolean mask derived from x_O selects only the occupied children,
          producing a compact list of child-level features.
    """
    # Occupancy mask: select only occupied children according to x_O
    occ_bits = _get_occ_bits()                   
    bits = occ_bits[x_O.view(-1).to(torch.long)]     
    mask = bits.view(-1)                             

    C = x_F.shape[1]
    x_F_rep = x_F.repeat(1, 8).reshape(-1, C)        
    x_up_F = x_F_rep[mask]                           

    return x_up_F

#####################################################################################
@torch.no_grad()
def upscale_coordinate(x_C: torch.Tensor, x_O: torch.Tensor) -> torch.Tensor:
    """
    Expands parent voxel coordinates to their occupied child voxel coordinates.

    Args:
        x_C: (N, 4) int tensor of parent voxel coordinates [b, x, y, z],
             where (x, y, z) are integer grid coordinates at a coarse level.
        x_O: (N,)   int tensor of occupancy symbols in [0, 255].
             Each symbol encodes which of the 8 children of a parent voxel
             are present:
                 bit 0: child (0,0,0)
                 bit 1: child (1,0,0)
                 bit 2: child (0,1,0)
                 bit 3: child (1,1,0)
                 bit 4: child (0,0,1)
                 bit 5: child (1,0,1)
                 bit 6: child (0,1,1)
                 bit 7: child (1,1,1)

    Returns:
        x_up_C: (N_up, 4) int tensor of child voxel coordinates [b, x, y, z]
                at the finer level, containing one row per occupied child voxel.

    Implementation details:
        1. Each parent coordinate (b, x, y, z) is upscaled to the finer level
           by multiplying spatial coordinates by 2:
               (x', y', z') = (2x, 2y, 2z)
        2. The 8 possible child offsets (0/1 in each axis) are stored in
           _get_expand_coords_base():
               [0,0,0], [1,0,0], [0,1,0], [1,1,0],
               [0,0,1], [1,0,1], [0,1,1], [1,1,1]
        3. For all parents, we first generate coordinates for all 8 children:
               (x_child, y_child, z_child) = (2x, 2y, 2z) + child_offset
        4. We then apply an occupancy mask derived from x_O (via _get_occ_bits),
           which selects only the coordinates of actually occupied child voxels.
    """
    N_d = x_C.shape[0]

    # 1-to-8 expansion of coordinates: all possible child offsets for each parent
    expand_coords_base = _get_expand_coords_base()        
    expand_coords = expand_coords_base.repeat(N_d, 1)          

    # Upscale parent coordinates to finer grid and replicate 8 times
    x_C_repeat = (x_C * 2).repeat(1, 8).reshape(-1, 4)          
    x_C_repeat[:, 1:] = x_C_repeat[:, 1:] + expand_coords      

    # Occupancy mask: select only occupied children according to x_O
    occ_bits = _get_occ_bits()                              
    bits = occ_bits[x_O.view(-1).to(torch.long)]                 
    mask = bits.view(-1)                                         

    x_up_C = x_C_repeat[mask].int()                                                               
    return x_up_C

#####################################################################################
@torch.no_grad()
def upscale_coordinate_feature(x_C: torch.Tensor, x_O: torch.Tensor, x_F: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expands parent voxel coordinates and features to their occupied child voxels.

    Args:
        x_C: (N, 4) int tensor of parent voxel coordinates [b, x, y, z],
             where (x, y, z) are integer grid coordinates at a coarse level.
        x_O: (N,)   int tensor of occupancy symbols in [0, 255].
             Each symbol encodes which of the 8 children of a parent voxel
             are present:
                 bit 0: child (0,0,0)
                 bit 1: child (1,0,0)
                 bit 2: child (0,1,0)
                 bit 3: child (1,1,0)
                 bit 4: child (0,0,1)
                 bit 5: child (1,0,1)
                 bit 6: child (0,1,1)
                 bit 7: child (1,1,1)
        x_F: (N, C) tensor of features at the parent (coarse) level.

    Returns:
        x_up_C: (N_up, 4) int tensor of child voxel coordinates [b, x, y, z]
                at the finer level, containing one row per occupied child voxel.
        x_up_F: (N_up, C) tensor of child-level features, where each occupied
                child inherits the feature of its parent.

    Implementation details:
        1. Each parent coordinate (b, x, y, z) is upscaled to the finer grid
           by multiplying spatial coordinates by 2:
               (x', y', z') = (2x, 2y, 2z)
        2. The 8 possible child offsets (0/1 per axis) are stored in
           _get_expand_coords_base():
               [0,0,0], [1,0,0], [0,1,0], [1,1,0],
               [0,0,1], [1,0,1], [0,1,1], [1,1,1]
        3. For all parents, we first generate coordinates and features for
           all 8 children:
               coords_child_all = (2 * coords_parent) + child_offset
               feats_child_all  = repeat(parent_feature, 8)
        4. We then derive a boolean occupancy mask from x_O (via _get_occ_bits)
           and use it to select only the coordinates and features of occupied
           child voxels.
    """
    N_d = x_C.shape[0]

    # 1-to-8 expansion of coordinates: all possible child offsets for each parent
    expand_coords_base = _get_expand_coords_base()        
    expand_coords = expand_coords_base.repeat(N_d, 1)          

    # Upscale parent coordinates to finer grid and replicate 8 times
    x_C_repeat = (x_C * 2).repeat(1, 8).reshape(-1, 4)          
    x_C_repeat[:, 1:] = x_C_repeat[:, 1:] + expand_coords      

    # Occupancy mask: select only occupied children according to x_O
    occ_bits = _get_occ_bits()                              
    bits = occ_bits[x_O.view(-1).to(torch.long)]                 
    mask = bits.view(-1)                                         
                
     # Expand features in the same 1-to-8 fashion and apply the same mask
    C = x_F.shape[1]
    x_F_rep = x_F.repeat(1, 8).reshape(-1, C)                
    
    x_up_C = x_C_repeat[mask].int()          
    x_up_F = x_F_rep[mask]                                   
    return x_up_C, x_up_F