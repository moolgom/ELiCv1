import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsparse import SparseTensor

import src.morton as mt
from src.nn import LocalFeatureNet, ResNetV1, OccGroupPredNet, OccGroupPriorNet, FOG, FCG


#########################################################################################################################################
# Coding Network
class ELiCv1_Core(nn.Module):
    def __init__(self, channels=32, kernel_size=3):
        super().__init__()
        
        self.local_net = LocalFeatureNet(channels)              # Octant positional embedding
        self.feat_resnet0 = ResNetV1(channels, kernel_size)     # Feature refinement before stage-1 coding
        self.feat_resnet1 = ResNetV1(channels, kernel_size)     # Feature refinement before stage-1 coding
        self.pred_net0 = OccGroupPredNet(channels)              # Stage-1 occupancy prediction
        self.pred_net1 = OccGroupPredNet(channels)              # Stage-2 occupancy prediction
        self.prior_net0 = OccGroupPriorNet(channels)            # Stage-1 occupancy context embedding
        self.prior_net1 = OccGroupPriorNet(channels)            # Stage-2 occupancy context embedding
        self.prior_s0_resnet0 = ResNetV1(channels, kernel_size) # Feature refinement after stage-1 coding
        self.prior_s0_resnet1 = ResNetV1(channels, kernel_size) # Feature refinement after stage-1 coding
        self.prior_s1_resnet0 = ResNetV1(channels, kernel_size) # Feature refinement after stage-2 coding
        self.prior_s1_resnet1 = ResNetV1(channels, kernel_size) # Feature refinement after stage-2 coding
        
        # Channel-wise blending weights between features propagated from level b-1 and current level b
        self.blend_weights = nn.Parameter(torch.full((2, channels), 1.0, dtype=torch.float32)) # [2, C]
        
    def forward(self, x_C, x_O_s0, x_O_s1, feats_prop=None):
        ##### Generate initial features #####
        x = SparseTensor(coords=x_C, feats=torch.ones((x_C.shape[0], 1), device=x_C.device))
        feats = self.local_net(x)
        
        ##### Fuse current & previous features #####
        if feats_prop is not None:
            w = torch.softmax(self.blend_weights, dim=0) # ensure w.sum() = 1, [2, C]
            w_curr, w_prev = w[0], w[1] # [C]
            feats = w_curr[None, :] * feats + w_prev[None, :] * feats_prop
        x.feats = feats
        
        ##### Refined features #####
        x = self.feat_resnet0(x) 
        x = self.feat_resnet1(x) 
        
        ##### 1st-stage prediction #####
        logits_s0 = self.pred_net0(x.feats)
        probs_s0 = F.softmax(logits_s0, dim=-1)
        
        ##### 2nd-stage prediction #####
        prior_s0 = self.prior_net0(x_O_s0)
        x.feats = x.feats + prior_s0
        x = self.prior_s0_resnet0(x)
        x = self.prior_s0_resnet1(x)
        logits_s1 = self.pred_net1(x.feats)
        probs_s1 = F.softmax(logits_s1, dim=-1)
        
        ##### For next bitdepth #####
        prior_s1 = self.prior_net1(x_O_s1)
        x.feats = x.feats + prior_s1
        x = self.prior_s1_resnet0(x)
        x = self.prior_s1_resnet1(x)
        
        ##### Total bits #####
        probs_s0 = probs_s0.gather(1, x_O_s0.unsqueeze(1).long()) 
        probs_s1 = probs_s1.gather(1, x_O_s1.unsqueeze(1).long()) 
        
        bits  = torch.sum(torch.clamp(-1.0 * torch.log2(probs_s0 + 1e-10), 0, 50))
        bits += torch.sum(torch.clamp(-1.0 * torch.log2(probs_s1 + 1e-10), 0, 50))
        
        return bits, x.feats
    


#########################################################################################################################################
# ELiC with BoE
class ELiCv1(nn.Module):
    def __init__(self, channels=32, kernel_size=3, K=4):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.K = K
        self.register_buffer('centers', torch.zeros(K, 32))  # BoE centers
        
        # Note: FOG and FCG from RENO were not used for training and testing.
        #       Added to compare Morton-order-preserving hierarchy with explicit per-level sorting.
        #       Safe to remove.
        self.fog = FOG()
        self.fcg = FCG()
        
        # Bag of the Encoders
        self.BoE = nn.ModuleList([
            ELiCv1_Core(channels, kernel_size)
            for _ in range(self.K + 1) # K+1 encoders (+1 is base coding network) 
        ])
        
    # Configure the parameters of all BoE networks similarly
    def copy_params_from_first(self, noise_std=0.0):
        with torch.no_grad():
            src_state = self.BoE[0].state_dict()
            for i in range(1, len(self.BoE)):
                self.BoE[i].load_state_dict(src_state)
                if noise_std > 0.0:
                    for p in self.BoE[i].parameters():
                        p.add_(torch.randn_like(p) * noise_std)
                        
        
    # Compare the input occupancy histogram with BoE centers to actively select the optimal encoder
    def select_encoder_index(self, x_O_s0, x_O_s1):
        x_H_s0 = torch.bincount(x_O_s0.int().view(-1), minlength=16)
        x_H_s1 = torch.bincount(x_O_s1.int().view(-1), minlength=16)
        x_H = torch.cat([x_H_s0, x_H_s1], dim=0).float()
        x_H = x_H / x_H.sum()
        dists = torch.norm(self.centers - x_H[None, :], dim=1) 
        idx = torch.argmin(dists).item()
        return idx
    
    
    # (Note) Assumes all batches share the same bitdepth
    def compute_bitdepth(self, coords):
        max_coord = coords[:, 1:].max().item() 
        bitdepth = math.ceil(math.log2(max_coord + 1))
        return bitdepth
    
    def forward(self, x):
        #####################################################################################
        num_in_points = x.coords.shape[0]
        multiscale_list = []
        
        bxyz, code = mt.morton3_sort(x.coords) # Initial morton sorting
        
        # Build morton-order preserving hierarchy and generate occupancy labels
        while True:
            bxyz, code, occ_sym = mt.down_once(bxyz, code) # Global morton-order-preserving downscaling
            multiscale_list.append((bxyz.clone(), occ_sym.clone()))
            curr_bitdepth = self.compute_bitdepth(bxyz)
            if curr_bitdepth <= 2:
                break
        multiscale_list = multiscale_list[::-1] 
        
        #####################################################################################
        total_bits = 0
        feats_prop = None
        curr_bitdepth = 2 # Start from b=2
        for depth in range(len(multiscale_list)):
            x_C, x_O = multiscale_list[depth]
            
            ##### GTs for two-stage coding #####
            x_O_s0 = x_O & 0xF   # Q_s1 == x_O % 16
            x_O_s1 = x_O >> 4    # Q_s2 == x_O // 16
            
            ##### Determine which encoder to use #####
            if curr_bitdepth <= 6: 
                enc_idx = self.K # Base coding network for b=2,3,4,5,6
            else:
                enc_idx = self.select_encoder_index(x_O_s0, x_O_s1) # Select the optimal encoder

            ##### Encode current bitdepth #####
            bits_curr, feats_curr = self.BoE[enc_idx](x_C, x_O_s0, x_O_s1, feats_prop)
            total_bits += bits_curr
            
            ##### For next bitdepth #####
            feats_prop = mt.upscale_feature(x_O, feats_curr) # Upscaling to propagate features to level b+1
            curr_bitdepth += 1

        bpp = total_bits / num_in_points
        return bpp
        