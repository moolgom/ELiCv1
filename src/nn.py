import torch
import torch.nn as nn
from torchsparse import nn as spnn


#########################################################################################################################################
# Fast Occupancy Generator from RENO
# https://github.com/NJUVISION/RENO
class FOG(torch.nn.Module): 
    """Fast Occupancy Converter
    """  
    def __init__(self):
        super(FOG, self).__init__()

        self.conv = spnn.Conv3d(1, 1, kernel_size=2, stride=2, bias=False)
        torch.nn.init.constant_(self.conv.kernel, 1.0)
        for param in self.conv.parameters():
            param.requires_grad = False

        self.pos_multiplier = torch.tensor([[1, 2, 4]], dtype=torch.int32, device='cuda')

    def pos(self, coords):
        """Assign codes (i.e., 1, 2, 4, ..., 64, 128) for each coords
        Input: coords: (N_d, 4)
        Return: pos: (N_d, 1)
        """ 
        pos = (coords[:, 1:] & 1) * self.pos_multiplier # (N_d, 3)
        pos = pos.sum(dim=-1, keepdim=True) # (N_d, 1)
        pos = (1 << pos).float()
        return pos

    def forward(self, x):
        # Dyadic downscaling to generate sparse occupancy code
        # Input: x SparseTensor: x.coords (N_d, 4); x.feats (N_d, 1)
        # Return: ds_x SparseTensor: ds_x.coords (N_{d-1}, 4); ds_x.feats (N_{d-1}, 1)
        
        x.feats = self.pos(x.coords) # (N{d-1}, 1)
        ds_x = self.conv(x) # coordinate = ds_x.C and occupancy = ds_x.F
        return ds_x
    
#########################################################################################################################################
# Fast Coordinate Generator from RENO
# https://github.com/NJUVISION/RENO

class FCG(torch.nn.Module): 
    """Fast Coordinate Converter
    """  
    def __init__(self):
        super(FCG, self).__init__()
        self.expand_coords_base = torch.tensor([
            [0, 0, 0], # -> 1 (occupancy adder)
            [1, 0, 0], # -> 2 (occupancy adder)
            [0, 1, 0], # -> 4 (occupancy adder)
            [1, 1, 0], # -> 8 (occupancy adder)
            [0, 0, 1], # -> 16 (occupancy adder)
            [1, 0, 1], # -> 32 (occupancy adder)
            [0, 1, 1], # -> 64 (occupancy adder)
            [1, 1, 1], # -> 128 (occupancy adder)
        ], dtype=torch.int32, device='cuda')

        self.pos = torch.arange(0, 8, device='cuda').view(1, 8)

    def forward(self, x_C, x_O, x_F=None):
        # Upscaling according to coordinates and occupancy code
        # Input: x_C: coordinates (N_d, 4)
        # Input: x_O: occupancy (N_d, 1)
        # Input: x_F: features (N_d, C)
        # Return: x_up_C: upscaled coordinates (N_{d+1}, 4)

        # 1 to 8 expand
        expand_coords = self.expand_coords_base.repeat(x_C.shape[0], 1) # (N_d*8, 4)
        x_C_repeat = x_C.repeat(1, 8).reshape(-1, 4) # (N_d*8, 4) repeated coords
        x_C_repeat[:, 1:] = x_C_repeat[:, 1:] * 2 + expand_coords # (N_d*8, 4) expanded coords
        mask = torch.div(x_O.repeat(1, 8) % (2**(self.pos+1)), 2**self.pos, rounding_mode='floor').reshape(-1) # (N_d*8, 1) mask for pruning
        mask = (mask == 1) # (N_d*8, 1) mask for pruning
        x_up_C = x_C_repeat[mask].int() # (N_{d+1}, 4) upscaled coords
        if x_F is None:
            return x_up_C
        else:
            C = x_F.shape[1]
            x_F = x_F.repeat(1, 8).reshape(-1, C) # (N_d*8, C)
            x_up_F = x_F[mask]
            return x_up_C, x_up_F
        
#########################################################################################################################################
# Octant Positional Embedding
class LocalFeatureNet(nn.Module): 
    def __init__(self, channels=32):
        super(LocalFeatureNet, self).__init__()
        self.channels = channels
        self.relative_position_embedding = nn.Embedding(8, channels)
        
    def forward(self, x):
        coords_abs = x.coords
        coords_rel = coords_abs[:, 1:] % 2
        idxs_rel = coords_rel[:, 0] + coords_rel[:, 1]*2 + coords_rel[:, 2]*4
        feats_pos = self.relative_position_embedding(idxs_rel.long()).reshape(-1, self.channels)
        
        return feats_pos
    
    
#########################################################################################################################################
# Plain Residual Block for Feature Refinement
class ResNetV1(nn.Module): 
    def __init__(self, channels=32, k=3):
        super(ResNetV1, self).__init__()
        
        self.conv0 = spnn.Conv3d(channels, channels, k)
        self.conv1 = spnn.Conv3d(channels, channels, k)
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out = self.relu(out+x)
        return out
    
#########################################################################################################################################
# Occupancy Prediction
class OccGroupPredNet(nn.Module):
    def __init__(self, channels=32):
        super(OccGroupPredNet, self).__init__()
        
        self.linear0 = nn.Linear(channels, channels)
        self.linear1 = nn.Linear(channels, 16)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.linear0(x)
        out = self.relu(out)
        logits = self.linear1(out)
        return logits
    
#########################################################################################################################################
# Occupancy Context Embedding
class OccGroupPriorNet(nn.Module):
    def __init__(self, channels=32):
        super(OccGroupPriorNet, self).__init__()
        self.channels = channels
        self.emb = nn.Embedding(16, channels)
        
    def forward(self, prior):
        ctx = self.emb(prior.long()).reshape(-1, self.channels)
        return ctx