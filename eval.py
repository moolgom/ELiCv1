import os
import argparse
import subprocess
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

from multiprocessing import Pool

parser = argparse.ArgumentParser(
    prog='eval.py',
    description='Eval geometry PSNR.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Ford
'''
parser.add_argument('--csv_filename', type=str, help='Glob pattern to load point clouds.', default='Ford_K5_posQ16.csv')
parser.add_argument('--pcc_metric_path', type=str, help='Path for pc_error_d.', default='./third_party/pc_error_d')
parser.add_argument('--resolution', type=float, help='Point cloud resolution (peak signal).', default=30000) 
'''
# KITTI

parser.add_argument('--csv_filename', type=str, help='Glob pattern to load point clouds.', default='KITTI_K5_posQ16.csv')
parser.add_argument('--pcc_metric_path', type=str, help='Path for pc_error_d.', default='./third_party/pc_error_d')
parser.add_argument('--resolution', type=float, help='Point cloud resolution (peak signal).', default=59.70) 

args = parser.parse_args()

df = pd.read_csv(args.csv_filename) 
in_files = df["in_file"].tolist()
out_files = df["out_file"].tolist()
out_files = [x.replace(".bin", ".ply") for x in out_files]

def process(file_pair):
    in_file, out_file = file_pair

    filename_wo_ext = os.path.split(in_file)[-1][:-4] # without extenstion
    cmd = f'{args.pcc_metric_path} \
    --fileA={in_file} --fileB={out_file} \
    --resolution={args.resolution} --inputNorm={in_file}'
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        d1_psnr = float(str(output).split('mseF,PSNR (p2point):')[1].split('\\n')[0])
        d2_psnr = float(str(output).split('mseF,PSNR (p2plane):')[1].split('\\n')[0])
    except (subprocess.CalledProcessError, IndexError, ValueError) as e:
        print('!!!Error!!!', cmd)
        print(e)
        d1_psnr, d2_psnr = -1, -1
        
    return np.array([filename_wo_ext, d1_psnr, d2_psnr])
    

input_pairs = list(zip(in_files, out_files))
with Pool(32) as p:
    arr = list(tqdm(p.imap(process, input_pairs), total=len(in_files)))        

arr = np.array(arr)
fnames, d1_PSNRs, d2_PSNRs = arr[:, 0], arr[:, 1].astype(float), arr[:, 2].astype(float)

print('Avg. D1 PSNR:', round(d1_PSNRs.mean(), 3))
print('Avg. D2 PSNR:', round(d2_PSNRs.mean(), 3))

sorted_idxs = np.argsort(fnames)
fnames_sorted = fnames[sorted_idxs]
d1_PSNRs_sorted = d1_PSNRs[sorted_idxs]
d2_PSNRs_sorted = d2_PSNRs[sorted_idxs]

df['D1 PSNR'] = d1_PSNRs_sorted.tolist()
df['D2 PSNR'] = d2_PSNRs_sorted.tolist() 
df.to_csv(args.csv_filename, index=False)