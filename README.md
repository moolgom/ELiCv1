# [CVPR 2026] ELiC: Efficient LiDAR Geometry Compression via Cross-Bit-depth Feature Propagation and Bag-of-Encoders

[![arXiv](https://img.shields.io/badge/Arxiv-2503.12382-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.14070)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE.md)

<p align="center">
  <img src="./imgs/ELiC.png" alt="ELiC Architecture" width="20%">
</p>

This repository is the offical PyTorch implementation of our paper *Efficient LiDAR Geometry Compression via Cross-Bit-depth Feature Propagation and Bag-of-Encoders*.

## 🎯 Abstract  

Hierarchical LiDAR geometry compression encodes voxel occupancies from low to high bit-depths, yet prior methods treat each depth independently and re-estimate local context from coordinates at every level, limiting compression efficiency. We present ELiC, a real-time framework that combines cross-bit-depth feature propagation, a Bag-of-Encoders (BoE) selection scheme, and a Morton-order-preserving hierarchy. Cross-bit-depth propagation reuses features extracted at denser, lower depths to support prediction at sparser, higher depths. 
BoE selects, per depth, the most suitable coding network from a small pool, adapting capacity to observed occupancy statistics without training a separate model for each level. The Morton hierarchy maintains global Z-order across depth transitions, eliminating per-level sorting and reducing latency. Together these components improve entropy modeling and computation efficiency, yielding state-of-the-art compression at real-time throughput on Ford and SemanticKITTI. 

## ⚙️ Usage

### ⚡ Quick Test

We provide a simple toy sample to quickly test the pipeline.
You can easily run the following scripts to perform compression, decompression, and evaluation:
```bash
python compress.py
python decompress.py
python eval.py
```

### 🗜️ Encoding

To encode LiDAR geometry sequence (e.g., from SemanticKITTI), use the `compress.py` script with the following options:

```bash
python compress.py \
  --input_glob=./here/input/sequence/*.ply \
  --output_folder=./here/output/directory/ \
  --is_data_pre_quantized=True \
  --posQ=16 \
  --channels=32 \
  --kernel_size=3 \
  --ckpt=./here/checkpoint/model.pt
```

*--input_glob*: Glob pattern to locate input .ply or .bin files.

*--output_folder*: Directory to save compressed output (e.g., .bin files).

*--is_data_pre_quantized*: Set True if input is already quantized (integer coordinates). Default: False, <span style="color:red">If False, do not specify this option.</span>

*--posQ*: Coordinate quantization scale. Allowed values: 4, 8, 16, 32, 64. Higher values → more compression, lower precision.

*--channels*: Number of neural network feature channels. Default: 32 (use 64 for ELiC-Large)

*--kernel_size*: Convolution kernel size. Default: 3

*--ckpt*: Path to pretrained model checkpoint.

### 🔄 Decoding


To decompress previously compressed `.bin` files into `.ply` format, run:

```bash
python decompress.py \
  --input_glob=./here/compressed/bin/*.bin \
  --output_folder=./here/output/directory/ \
  --is_data_pre_quantized=True \
  --channels=32 \
  --kernel_size=3 \
  --ckpt=./here/checkpoint/model.pt
``` 

*--input_glob*: Glob pattern to locate compressed .bin files.

*--output_folder*: Directory to save compressed output (e.g., .ply files).
 
*--is_data_pre_quantized*: Set True if input is already quantized (integer coordinates). Default: False, <span style="color:red">If False, do not specify this option.</span>

*--channels*: Number of neural network feature channels. Default: 32 (use 64 for ELiC-Large)

*--kernel_size*: Convolution kernel size. Default: 3

*--ckpt*: Path to pretrained model checkpoint.

### 🧠 Training

Before training ELiC, the Bag-of-Encoders (BoE) centers must be computed first:

```bash
python BoE_cluster.py \
  --train_data=./here/training/dataset/*.bin \
  --model_save_folder=./here/model/directory/ \
  --K=5 \
  --is_data_pre_quantized=True 
```

*--train_data*: Glob pattern pointing to the training dataset.

*--model_save_folder*: Directory for saving BoE centers after clustering.
<span style="color:red">Should be set to the same location where ELiC checkpoints will later be stored.</span>

*--K*: Number of clusters for K+1 BoE pool. (excluding base coding network)

*--is_data_pre_quantized*: Set True if training dataset is already quantized (integer coordinates). Default: False, <span style="color:red">If False, do not specify this option.</span>


To train the ELiC model, use the following command:

```bash
python train.py \
  --train_data=./here/training/dataset/*.bin \
  --model_save_folder=./here/model/directory/ \
  --K=5 \
  --ckpt_name=ELiCv1 \
  --is_data_pre_quantized=True \
  --channels=32 \
  --kernel_size=3 \
  --batch_size=1 \
  --learning_rate=0.0005 \
  --lr_decay=0.1 \
  --lr_decay_steps=150000 250000 \
  --max_steps=300000
```

*--train_data*: Glob pattern pointing to the training dataset.

*--model_save_folder*: Directory to save model checkpoints after training.

*--K*: Number of clusters for K+1 BoE pool. (excluding base coding network)

*--ckpt_name*: Prefix used when saving the model checkpoint files.

*--is_data_pre_quantized*: Set True if training dataset is already quantized (integer coordinates). Default: False, <span style="color:red">If False, do not specify this option.</span>

*--channels*: Number of feature channels in the model. Default: 32 (Use 64 for ELiC-Large)

*--kernel_size*: Convolution kernel size. Default: 3

*--batch_size*: Number of samples per training batch. Default: 1 (Currently, only supporting batch_size=1)

*--learning_rate*: Initial learning rate for training. Default: 0.0005

*--lr_decay*: Multiplicative factor for learning rate decay. Default: 0.1 (i.e., LR becomes 10% at each decay step)

*--lr_decay_steps*: Steps at which learning rate decays. Example: 150000 250000

*--max_steps*: Total number of training iterations. Default: 300000



### ⚖️ License

This project is licensed under the [BSD 3-Clause License](LICENSE.md).

The license applies only to the software provided in this repository.
No patent rights are granted under this license. Technologies described
in this repository may be covered by patents owned by the Electronics
and Telecommunications Research Institute (ETRI). Commercial use of such
patented technologies may require a separate patent license from ETRI.


## 🙏 Acknowledgements

Parts of this codebase are inspired by the implementation of

RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds  
https://github.com/NJUVISION/RENO

We appreciate the authors for making their code publicly available.
Please refer to the original repository for the baseline implementation.

<!--
---

> 📌 **Note**  
> This README is a work in progress and will be continuously updated.  
> Please check back later for more detailed instructions, extended usage examples, and additional dataset support.
-->

