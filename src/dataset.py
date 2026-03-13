import math
import numpy as np
from glob import glob

import torch

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn

import src.io as io 

#########################################################################################################################################
class PCDataset:
    def __init__(self, file_path_ls, posQ=4, is_pre_quantized=False, augment_data=True):
        self.files = io.read_point_clouds(file_path_ls)
        self.posQ = posQ
        self.is_pre_quantized = is_pre_quantized
        self.augment_data = augment_data

    def __len__(self):
        return len(self.files)

    def rotate_z(self, xyz, angle_rad):
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        
        R = torch.tensor([
            [cos_theta, -sin_theta, 0.0],
            [sin_theta,  cos_theta, 0.0],
            [0.0,        0.0,       1.0]
        ], dtype=xyz.dtype, device=xyz.device)
        
        return xyz @ R.T  # shape: (N, 3)
    
    def __getitem__(self, idx):
        xyz = torch.tensor(self.files[idx], dtype=torch.float)
        
        if self.augment_data:
            angle = torch.rand(1).item() * 2 * math.pi  # 0 ~ 2pi random rotation
            xyz = self.rotate_z(xyz, angle)
        
        feats = torch.ones((xyz.shape[0], 1), dtype=torch.float)
        
        if not self.is_pre_quantized:
            xyz = xyz / 0.001 
        xyz = torch.round((xyz + 131072) / self.posQ).int()

        input = SparseTensor(coords=xyz, feats=feats)
        
        return {"input": input}


#########################################################################################################################################
def get_data_loader(dataset, is_data_pre_quantized, batch_size, augment_data=True):
    files = np.array(glob(dataset, recursive=True))
    np.random.shuffle(files)
    
    dataloader = torch.utils.data.DataLoader(
        dataset=PCDataset(files, is_pre_quantized=is_data_pre_quantized, augment_data=augment_data),
        shuffle=True,
        batch_size=batch_size,
        collate_fn=sparse_collate_fn
    )
    return dataloader
