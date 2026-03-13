import os
import re
import time
import argparse
import pandas as pd

import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torchsparse import SparseTensor
from torchsparse.nn import functional as F

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
ac_dec = ac_ext.ArithmeticDecoder16()

# Set torchsparse config
conv_config = F.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(conv_config)

#########################################################################################################################################
parser = argparse.ArgumentParser(
    prog='decompress.py',
    description='Decompress point cloud geometry.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
# Ford
'''
parser.add_argument('--input_glob', default='./data/Ford/*.bin', help='Glob pattern for input bin files.')
parser.add_argument('--output_folder', default='./data/Ford/', help='Folder to save decompressed ply files.')
parser.add_argument("--is_data_pre_quantized", type=bool, default=True, help="Whether the input data is pre quantized.")
parser.add_argument('--channels', type=int, help='Neural network channels.', default=32)
parser.add_argument('--kernel_size', type=int, help='Convolution kernel size.', default=3)
parser.add_argument('--ckpt', help='Checkpoint load path.', default='./model/Ford/ELiCv1_K5.pt')
'''
# SemanticKITTI

parser.add_argument('--input_glob', default='./data/SemanticKITTI/*.bin', help='Glob pattern for input bin files.')
parser.add_argument('--output_folder', default='./data/SemanticKITTI/', help='Folder to save decompressed ply files.')
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether the input data is pre quantized.")
parser.add_argument('--channels', type=int, help='Neural network channels.', default=32)
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

dec_time_ls = []

with torch.no_grad():
    for file_path in tqdm(in_file_path_ls):
        file_name = os.path.split(file_path)[-1]
        decompressed_file_path = os.path.join(args.output_folder, file_name[:-4]+'.ply')
        
        with open(file_path, 'rb') as f:
            posQ = np.frombuffer(f.read(2), dtype=np.int16)[0]
            base_x_len = np.frombuffer(f.read(1), dtype=np.int8)[0]
            base_x_coords = np.frombuffer(f.read(base_x_len*3), dtype=np.int8).astype(np.int32)
            byte_stream = f.read()
            bitstream, enc_idx_list = op.unpack_bitstream_and_enc_idx_list(byte_stream)
        ################################ Decompress
        
        base_x_coords = torch.tensor(base_x_coords.reshape(-1, 3), device=device) 
        base_x_coords = torch.cat((base_x_coords[:, 0:1]*0*0, base_x_coords), dim=-1).to(device)
        
        x_C = base_x_coords
        ac_dec.reset(bitstream)
        with torch.inference_mode():
            
            torch.cuda.synchronize()
            dec_time_start = time.time()
            #####################################################################################
            feats_prop = None
            curr_bitdepth = 2
            
            for depth, enc_idx in enumerate(enc_idx_list):
                last_depth = (depth == len(enc_idx_list) - 1)
            
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
                probs_s0 = torch.nn.functional.softmax(logits_s0, dim=-1)
                
                ##### 1st-stage Arithmetic Decoding #####
                probs_cdf_s0 = torch.cat((probs_s0[:, 0:1]*0, probs_s0.cumsum(dim=-1)), dim=-1)
                probs_cdf_norm_s0 = op._convert_to_int_and_normalize(probs_cdf_s0, True)
                probs_cdf_norm_s0 = probs_cdf_norm_s0.cpu()
                x_O_s0 = ac_dec.decode_chunk(probs_cdf_norm_s0).cuda()
                
                ##### 2nd-stage prediction #####
                x.feats += net.BoE[enc_idx].prior_net0(x_O_s0)
                x = net.BoE[enc_idx].prior_s0_resnet0(x)
                x = net.BoE[enc_idx].prior_s0_resnet1(x)
                logits_s1 = net.BoE[enc_idx].pred_net1(x.feats)
                probs_s1 = torch.nn.functional.softmax(logits_s1, dim=-1)
                    
                ##### 2nd-stage Arithmetic Decoding #####
                probs_cdf_s1 = torch.cat((probs_s1[:, 0:1]*0, probs_s1.cumsum(dim=-1)), dim=-1)
                probs_cdf_norm_s1 = op._convert_to_int_and_normalize(probs_cdf_s1, True)
                probs_cdf_norm_s1 = probs_cdf_norm_s1.cpu()
                x_O_s1 = ac_dec.decode_chunk(probs_cdf_norm_s1).cuda()

                ##### Merge two-stage predictions #####
                x_O = x_O_s1 * 16 + x_O_s0
                
                ##### For next bitdepth #####
                if not last_depth:  # If not final depth
                    x.feats += net.BoE[enc_idx].prior_net1(x_O_s1)
                    x = net.BoE[enc_idx].prior_s1_resnet0(x)
                    x = net.BoE[enc_idx].prior_s1_resnet1(x)
                    x_C, feats_prop = mt.upscale_coordinate_feature(x_C, x_O, x.feats) # Global morton-order-preserving upscaling
                    curr_bitdepth += 1
                else: # final depth
                    x_C = mt.upscale_coordinate(x_C, x_O) # Global morton-order-preserving upscaling
            #####################################################################################
            torch.cuda.synchronize()
            dec_time_end = time.time()
            dec_time = dec_time_end-dec_time_start
            scan = x_C
        
        if args.is_data_pre_quantized:
            scan = scan[:, 1:] * posQ  - 131072
        else:
            scan = (scan[:, 1:] * posQ - 131072) * 0.001
            
        dec_time_ls.append(dec_time)
        io.save_ply_ascii_geo(scan.float().cpu().numpy(), decompressed_file_path)
        
print('Total: {total_n:d} | Decode Time:{dec_time:.3f} | Max GPU Memory:{memory:.2f}MB'.format(
    total_n=len(dec_time_ls),
    dec_time=np.array(dec_time_ls[1:-1]).mean(),
    memory=torch.cuda.max_memory_allocated()/1024/1024
))

# output csv file
testset = "Unknown"
if "Ford" in args.input_glob:
    testset = "Ford"
elif "KITTI" in args.input_glob:
    testset = "KITTI"
csv_filename = f"{testset}_K{K}_posQ{int(posQ)}.csv"

df = pd.read_csv(csv_filename)
df['dec_time'] = dec_time_ls 
df.to_csv(csv_filename, index=False)
