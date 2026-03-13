import os
import re
import argparse
import pandas as pd

import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchsparse import SparseTensor
from torchsparse.nn import functional as spnnF

import src.io as io
import src.op as op
import src.morton as mt
from model import ELiCv1

# Arithmetic Coder
from torch.utils.cpp_extension import load
_this_dir = os.path.dirname(os.path.abspath(__file__))
ac_ext = load(
    name="ac_ext",  
    sources=[os.path.join(_this_dir, "csrc/arithmetic_coding.cpp")],
    extra_cflags=["-O3"],
    with_cuda=False,
    verbose=True,
)
ac_enc = ac_ext.ArithmeticEncoder16()

# Set torchsparse config
conv_config = spnnF.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
spnnF.conv_config.set_global_conv_config(conv_config)

#########################################################################################################################################
parser = argparse.ArgumentParser(
    prog='compress.py',
    description='Compress point cloud geometry.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Ford
'''
parser.add_argument('--input_glob', default='./data/FordExample/*.ply', help='Glob pattern for input point clouds.')
parser.add_argument('--output_folder', default='./data/Ford/', help='Folder to save compressed bin files.')
parser.add_argument("--is_data_pre_quantized", type=bool, default=True, help="Whether the input data is pre quantized.")
parser.add_argument('--posQ', default=16, type=int, help='Quantization scale.')

parser.add_argument('--channels', type=int, help='Neural network channels.', default=32)  # 64 for ELiC-Large
parser.add_argument('--kernel_size', type=int, help='Convolution kernel size.', default=3)
parser.add_argument('--ckpt', help='Checkpoint load path.', default='./model/Ford/ELiCv1_K5.pt')
'''
# SematicKITTI

parser.add_argument('--input_glob', default='./data/SemanticKITTIExample/*.ply', help='Glob pattern for input point clouds.')
parser.add_argument('--output_folder', default='./data/SemanticKITTI/', help='Folder to save compressed bin files.')
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether the input data is pre quantized.") # Do not use this option when set to False.
parser.add_argument('--posQ', default=16, type=int, help='Quantization scale.')

parser.add_argument('--channels', type=int, help='Neural network channels.', default=32)  # 64 for ELiC-Large
parser.add_argument('--kernel_size', type=int, help='Convolution kernel size.', default=3)
parser.add_argument('--ckpt', help='Checkpoint load path.', default='./model/SemanticKITTI/ELiCv1_K5.pt')

#########################################################################################################################################
args = parser.parse_args()

# Create model save path
os.makedirs(args.output_folder, exist_ok=True)

op.set_seed(seed=11)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_file_path_ls = glob(args.input_glob, recursive=True)
in_file_path_ls = sorted(in_file_path_ls)
# reading point cloud using multithread
xyz_ls = io.read_point_clouds(in_file_path_ls)

match = re.search(r'_K(\d+)', args.ckpt)
if match:
    K = int(match.group(1))
else:
    print("K not found in checkpoint filename.")

# Network
net = ELiCv1(channels=args.channels, kernel_size=args.kernel_size, K=K).to(device)
net.load_state_dict(torch.load(args.ckpt))
net.cuda().eval()

# Warm-up
random_coords = torch.randint(low=0, high=4096, size=(4096, 3)).int().to(device)
net(SparseTensor(coords=torch.cat((random_coords[:, 0:1]*0, random_coords), dim=-1),
                feats=torch.ones((4096, 1))).to(device))

# Pre-compute Cross-Bit-depth Feature Blending weights
with torch.no_grad():
    for enc_idx in range(net.K + 1):
        p = net.BoE[enc_idx].blend_weights           
        p.copy_(torch.softmax(p, dim=0))          
        
# output csv file
testset = "Unknown"
if "Ford" in args.input_glob:
    testset = "Ford"
elif "KITTI" in args.input_glob:
    testset = "KITTI"
csv_filename = f"{testset}_K{K}_posQ{args.posQ}.csv"

enc_time_ls, cpu_time_ls, bpp_ls = [], [], []
out_file_path_ls = []

start_gpu = torch.cuda.Event(enable_timing=True)
end_gpu = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    for file_idx in tqdm(range(len(in_file_path_ls))):
        file_path = in_file_path_ls[file_idx]
        file_name = os.path.split(file_path)[-1]
        compressed_file_path = os.path.join(args.output_folder, file_name[:-4]+'.bin')

        ################################ Get xyz
        if args.is_data_pre_quantized:
            xyz = torch.tensor(xyz_ls[file_idx])
            if xyz.min().item() < 0:
                xyz += 131072
        else:
            xyz = torch.tensor(xyz_ls[file_idx] / 0.001 + 131072)
        
        xyz = torch.round(xyz / args.posQ).int()
        bxyz = torch.cat((xyz[:,0:1]*0, xyz), dim=-1).int().to(device)
        N = xyz.shape[0]

        with torch.inference_mode():
            
            cpu_time = 0.0
            start_gpu.record()
            #####################################################################################
            multiscale_list = []
            enc_idx_list = []
            feats_prop = None
            
            bxyz, code = mt.morton3_sort(bxyz) # Initial morton sorting
            
            while True:
                bxyz, code, occ_sym = mt.down_once(bxyz, code) # Global morton-order-preserving downscaling
                multiscale_list.append((bxyz.clone(), occ_sym.clone()))
                curr_bitdepth = net.compute_bitdepth(bxyz)
                if curr_bitdepth <= 2:
                    break
            multiscale_list = multiscale_list[::-1] # reverse list
            
            curr_bitdepth = 2
            for depth in range(len(multiscale_list)):
                last_depth = (depth == len(multiscale_list) - 1)
                
                x_C, x_O = multiscale_list[depth]
               
                ##### GTs for two-stage coding #####
                x_O_s0 = x_O & 0xF   # == x_O % 16
                x_O_s1 = x_O >> 4    # == x_O // 16
                
                ##### Determine which encoder to use #####
                if curr_bitdepth <= 6:
                    enc_idx = net.K # Base coding network 
                else:
                    enc_idx = net.select_encoder_index(x_O_s0, x_O_s1) # Adaptively selected coding network
                    
                x = SparseTensor(coords=x_C, feats=torch.ones((x_C.shape[0], 1), device=x_C.device))
                x.feats = net.BoE[enc_idx].local_net(x)
                
                ##### Fuse current & previous features #####
                if feats_prop is not None:
                    x.feats = net.BoE[enc_idx].blend_weights[0] * x.feats + \
                        net.BoE[enc_idx].blend_weights[1] * feats_prop
                        
                ##### Refined features #####
                x = net.BoE[enc_idx].feat_resnet0(x) 
                x = net.BoE[enc_idx].feat_resnet1(x) 
                
                ##### 1st-stage prediction #####
                logits_s0 = net.BoE[enc_idx].pred_net0(x.feats)
                probs_s0 = F.softmax(logits_s0, dim=-1)
                
                ##### 2nd-stage prediction #####
                x.feats += net.BoE[enc_idx].prior_net0(x_O_s0)
                x = net.BoE[enc_idx].prior_s0_resnet0(x)
                x = net.BoE[enc_idx].prior_s0_resnet1(x)
                logits_s1 = net.BoE[enc_idx].pred_net1(x.feats)
                probs_s1 = F.softmax(logits_s1, dim=-1)
                    
                ##### Arithmetic Encoding #####
                probs_all = torch.cat((probs_s0, probs_s1), dim=0)
                probs_cdf = torch.cat((probs_all[:, 0:1]*0, probs_all.cumsum(dim=-1)), dim=-1) # (Nt, 257)
                probs_cdf_norm = op._convert_to_int_and_normalize(probs_cdf, True)

                probs_cdf_norm = probs_cdf_norm.cpu()
                half_num_gt_occ = x_O_s0.shape[0]
                ac_enc.encode_chunk(probs_cdf_norm[:half_num_gt_occ], x_O_s0.to(torch.int16).cpu())
                ac_enc.encode_chunk(probs_cdf_norm[half_num_gt_occ:], x_O_s1.to(torch.int16).cpu())
                enc_idx_list.append(enc_idx)
                
                ##### For next bitdepth #####
                if not last_depth:  # If not final depth
                    x.feats += net.BoE[enc_idx].prior_net1(x_O_s1)
                    x = net.BoE[enc_idx].prior_s1_resnet0(x)
                    x = net.BoE[enc_idx].prior_s1_resnet1(x)
                    feats_prop = mt.upscale_feature(x_O, x.feats) # Global morton-order-preserving upscaling
                    curr_bitdepth += 1
            
            #####################################################################################
            bitstream = ac_enc.finish()
            end_gpu.record()
            torch.cuda.synchronize()
            enc_time = start_gpu.elapsed_time(end_gpu) / 1000.0 # ms -> sec
        # End of with torch.inference_mode():
        
        base_x_coords, base_x_feats = multiscale_list[0]
        base_x_len = base_x_coords.shape[0] 
        base_x_coords = base_x_coords[:, 1:].cpu().numpy() # (n, 3)
        byte_stream = op.pack_bitstream_and_enc_idx_list(bitstream, enc_idx_list)
        
        with open(compressed_file_path, 'wb') as f:
            f.write(np.array(args.posQ, dtype=np.int16).tobytes())
            f.write(np.array(base_x_len, dtype=np.int8).tobytes())
            f.write(np.array(base_x_coords, dtype=np.int8).tobytes())
            f.write(byte_stream)
            
        enc_time_ls.append(enc_time)
        cpu_time_ls.append(cpu_time)
        bpp_ls.append(op.get_file_size_in_bits(compressed_file_path)/N)
        out_file_path_ls.append(compressed_file_path)


df = pd.DataFrame({
    'in_file': in_file_path_ls,
    'out_file': out_file_path_ls,
    'bpp': bpp_ls,
    'enc_time': enc_time_ls,
})
df.to_csv(csv_filename, index=False)

print('Total: {total_n:d} | Avg. Bpp:{bpp:.4f} | Encode time:{enc_time:.3f} | Max GPU Memory:{memory:.2f}MB'.format(
    total_n=len(enc_time_ls),
    bpp=np.array(bpp_ls).mean(),
    enc_time=np.array(enc_time_ls).mean(),
    memory=torch.cuda.max_memory_allocated()/1024/1024
))
