import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import torch

from torchsparse.nn import functional as F

import src.op as op
import src.morton as mt
from src.dataset import get_data_loader

# Set torchsparse config
conv_config = F.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(conv_config)

##############################################################################
def compute_bitdepth(coords, device):
    max_coord = coords[:, 1:].max().item() 
    bitdepth = math.ceil(math.log2(max_coord + 1))
    return torch.tensor(bitdepth, dtype=torch.long, device=device)    

##############################################################################
def run_kmeans(data, K=4):
    data_np = data.detach().cpu().numpy().astype('float32')
    kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto', verbose=1)
    kmeans.fit(data_np)
    centers = torch.from_numpy(kmeans.cluster_centers_)
    
    labels = kmeans.labels_
    for i in range(K):
        count = np.sum(labels == i)
        print(f'Cluster {i}: {count} samples')
    return centers

##############################################################################
def visualize_and_save(data, centers, bitdepths, save_path='kmeans_tsne.png'):
    all_data = torch.cat([data, centers.to(data.device)], dim=0).cpu().numpy()
    bitdepths_np = bitdepths.view(-1).cpu().numpy()
    
    n_data = data.shape[0]
    n_centers = centers.shape[0]
    
    # t-SNE to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embedding = tsne.fit_transform(all_data)

    # Split
    embedded_data = embedding[:n_data]
    embedded_centers = embedding[n_data:]

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedded_data[:, 0], embedded_data[:, 1], 
                          c=bitdepths_np, cmap='viridis', s=10, alpha=0.6, label='Distributions')
    plt.scatter(embedded_centers[:, 0], embedded_centers[:, 1], 
                s=100, c='red', marker='x', label='Cluster Centers')
    cbar = plt.colorbar(scatter)
    cbar.set_label("Bit-depth")
    plt.legend()
    plt.title('t-SNE of Histogram Distributions and Cluster Centers')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

#############################################################################################################
def main():
    parser = argparse.ArgumentParser()
    # Ford
    parser.add_argument('--train_data', default='/workspace/Dataset/LiDAR/Ford/Ford_01_q_1mm/*.ply', help='Training data (Glob pattern).')
    parser.add_argument('--model_save_folder', default='./model/Ford', help='Directory where to save trained models.')
    parser.add_argument('--K', type=int, default=5, help='Number of clusters to use in K-means.')
    parser.add_argument("--is_data_pre_quantized", type=bool, default=True, help="Whether the training data is pre quantized.")
    
    # SematicKITTI
    '''
    parser.add_argument('--train_data', default='/workspace/Dataset/LiDAR/SemanticKITTI/00-10/*.bin', help='Training data (Glob pattern).')
    parser.add_argument('--model_save_folder', default='./model/SematicKITTI', help='Directory where to save trained models.')
    parser.add_argument('--K', type=int, default=5, help='Number of clusters to use in K-means.')
    parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether the training data is pre quantized.")  # Do not use this option when set to False.
    '''
    #########################################################################################################################################
    args = parser.parse_args()
    
    # Create model save path
    os.makedirs(args.model_save_folder, exist_ok=True)
   
    op.set_seed(seed=11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_data_loader(args.train_data, args.is_data_pre_quantized, 1, augment_data=False)
    
    num_tr_data = len(train_loader)
    max_samples = num_tr_data * 9 # For each frame, collect histograms for 15, 14, 13, 12, 11, 10, 9, 8, 7 bit-depths
    cnt_samples = 0
    
    # Histogram information of the occupancy labels to be predicted (16 + 16 = 32)
    x_Hs = torch.zeros((max_samples, 32), dtype=torch.float32, device=device)
    # Bit-depths 
    x_Bs = torch.zeros((max_samples, 1),  dtype=torch.int32, device=device) 
    while cnt_samples < max_samples:
        for tr_data in train_loader:
            x = tr_data['input'].to(device=device)
            
            multiscale_list = []
            bxyz, code = mt.morton3_sort(x.coords) # Initial morton sorting
            while True:
                bxyz, code, occ_sym = mt.down_once(bxyz, code) # Global morton-order-preserving downscaling
                multiscale_list.append((bxyz.clone(), occ_sym.clone()))
                bitdepth = compute_bitdepth(bxyz, device)
                if bitdepth <= 7: # Collect data only up to bit-depth 7
                    break
                
            multiscale_list = multiscale_list[::-1]
            bitdepth = 7
            for depth in range(len(multiscale_list)):
                x_C, x_O = multiscale_list[depth]
                x_C, x_O = op.sort_CF(x_C, x_O)
                
                x_O_s0 = torch.remainder(x_O, 16) # 8-4-2-1
                x_O_s1 = torch.div(x_O, 16, rounding_mode='floor') # 128-64-32-16
                x_H_s0 = torch.bincount(x_O_s0.int().view(-1), minlength=16)
                x_H_s1 = torch.bincount(x_O_s1.int().view(-1), minlength=16)
                x_H = torch.cat([x_H_s0, x_H_s1], dim=0).float()
                x_H = x_H / x_H.sum()
                
                x_Hs[cnt_samples, :] = x_H
                x_Bs[cnt_samples, :] = bitdepth
                bitdepth += 1
                cnt_samples += 1
                
                if cnt_samples >= max_samples:
                    break
            if cnt_samples >= max_samples:
                break
            
    x_Hs = x_Hs[:cnt_samples, :]
    x_Bs = x_Bs[:cnt_samples, :]
    
    # Perform clustering on occupancy histogram information for bit-depths 7 to 15
    # so that the optimal coding network can be selected using the cluster centroids
    centers = run_kmeans(x_Hs, K=args.K)
    np.save(os.path.join(args.model_save_folder, f"Centers_{args.K}.npy"), centers.cpu().numpy())
    visualize_and_save(x_Hs, centers, x_Bs, save_path=os.path.join(args.model_save_folder, f"Centers_{args.K}.png"))

#############################################################################################################
if __name__ == '__main__':
    main()
    
    