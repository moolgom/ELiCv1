import os
import datetime
import argparse
import numpy as np

import torch
from torch.cuda import amp
from torchsparse.nn import functional as F

import src.op as op
from src.dataset import get_data_loader
from model import ELiCv1


# Set torchsparse config
conv_config = F.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(conv_config)


#############################################################################################################
def main():
    parser = argparse.ArgumentParser()
    # Ford
    '''
    parser.add_argument('--train_data', default='/Ford/Ford_01_q_1mm/*.ply', help='Training data (Glob pattern).')
    parser.add_argument('--model_save_folder', default='./model/Ford', help='Directory where to save trained models.')
    parser.add_argument('--K', type=int, default=5, help='Number of clusters to use in K-means.')
    parser.add_argument('--ckpt_name', default='ELiCv1', help='Prefix name used when saving trained model checkpoints.')
    parser.add_argument("--is_data_pre_quantized", type=bool, default=True, help="Whether the training data is pre quantized.")

    parser.add_argument('--channels', type=int, help='Neural network channels.', default=32) # 64 for ELiC-Large
    parser.add_argument('--kernel_size', type=int, help='Convolution kernel size.', default=3)

    parser.add_argument('--batch_size', type=int, help='Batch size.', default=1)
    parser.add_argument('--learning_rate', type=float, help='Learning rate.', default=0.0005)
    parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
    parser.add_argument('--lr_decay_steps', help='Decays the learning rate at x steps.', default=[150000, 250000])
    parser.add_argument('--max_steps', type=int, help='Train up to this number of steps.', default=300000)
    '''
    # SematicKITTI

    parser.add_argument('--train_data', default='/SemanticKITTI/00-10/*.bin', help='Training data (Glob pattern).')
    parser.add_argument('--model_save_folder', default='./model/SemanticKITTI', help='Directory where to save trained models.')
    parser.add_argument('--K', type=int, default=5, help='Number of BoE clusters to use in K-means.')
    parser.add_argument('--ckpt_name', default='ELiCv1', help='Prefix name used when saving trained model checkpoints.') 
    parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether the training data is pre quantized.") # Do not use this option when set to False.

    parser.add_argument('--channels', type=int, help='Neural network channels.', default=32) # 64 for ELiC-Large
    parser.add_argument('--kernel_size', type=int, help='Convolution kernel size.', default=3)

    parser.add_argument('--batch_size', type=int, help='Batch size.', default=1)
    parser.add_argument('--learning_rate', type=float, help='Learning rate.', default=0.0005)
    parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
    parser.add_argument('--lr_decay_steps', help='Decays the learning rate at x steps.', default=[150000, 250000])
    parser.add_argument('--max_steps', type=int, help='Train up to this number of steps.', default=300000)

    #########################################################################################################################################
    args = parser.parse_args()
     
    # Create model save path
    os.makedirs(args.model_save_folder, exist_ok=True)
   
    op.set_seed(seed=np.random.randint(1000))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training dataset loader
    train_loader = get_data_loader(args.train_data, args.is_data_pre_quantized, args.batch_size)
    
    # ELiC model
    net = ELiCv1(channels=args.channels, kernel_size=args.kernel_size, K=args.K).to(device).train()
    net.copy_params_from_first()
    
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scaler = amp.GradScaler(enabled=True)

    # BoE centers
    centers = np.load(os.path.join(args.model_save_folder, f"Centers_{args.K}.npy"))
    centers = torch.from_numpy(centers).to(device)
    net.centers.copy_(centers) # Store the BoE centers in the model
    
    # Use RENO's training loop as is
    losses = []
    global_step = 0
    for epoch in range(1, 9999):
        print(datetime.datetime.now())
        net.train()
        for tr_data in train_loader:
            x = tr_data['input'].to(device=device)
            
            with amp.autocast(enabled=True):
                loss = net(x)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

            # PRINT
            if global_step % 500 == 0: 
                print(f'Epoch:{epoch} | Step:{global_step} | Loss:{round(np.array(losses).mean(), 5)}')
                losses = []

            # LEARNING RATE DECAY
            if global_step in args.lr_decay_steps:
                args.learning_rate = args.learning_rate * args.lr_decay
                for g in optimizer.param_groups:
                    g['lr'] = args.learning_rate
                print(f'Learning rate decay triggered at step {global_step}, LR is setting to {args.learning_rate}.')
                
            if global_step >= args.max_steps:
                break
            
            global_step += 1

        # Save model
        net.eval()
        torch.save(net.state_dict(), os.path.join(args.model_save_folder, f"{args.ckpt_name}_K{args.K}.pt"))

        if global_step >= args.max_steps:
            break

#############################################################################################################
if __name__ == '__main__':
    main()
    