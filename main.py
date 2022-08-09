import os
from xml.etree.ElementInclude import default_loader
import torch
import argparse
from torch.backends import cudnn
from net.Unet import Unet
from net.UNetplusplus import UNetplusplus
from net.U2Net import U2NETP
from net.PMRID import PMRID
from net.PMRID_ca import PMRID_ca
from net.PMRID_caplus import PMRID_caplus
from net.PMRID_caplusp import PMRID_caplusp
from net.MPRNet import MPRNet
from train import _train
from test import _test



def main(args):
    # CUDNN
    # cudnn.benchmark = True

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    model = PMRID_caplusp().to(args.device)
    torch.cuda.empty_cache()
    
    if args.mode == 'train':
       _train(model, args) # for img_split
       #_train(model, args) # for img_total

    elif args.mode == 'test':
        _test(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='PMRID_caplusp',  type=str)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4) # default=1e-4
    parser.add_argument('--weight_decay', type=float, default=0) ## default=0, 5e-2
    parser.add_argument('--num_epoch', type=int, default=999)
    parser.add_argument('--device', type=str, default = 'cuda:7' )
    # parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--num_worker', type=int, default=2)
    # parser.add_argument('--save_freq', type=int, default=10)
    # parser.add_argument('--valid_freq', type=int, default=10)
    # parser.add_argument('--ckp', type=str, default='checkpoints/best_model.pth') #'../checkpoints/pretrained/pretrained.pth')
    # parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+2) * 40 for x in range(999//50)])
    parser.add_argument('--test_dir', type=str, default='data/valid/noisy/')

    args = parser.parse_args()
    args.model_save_dir = os.path.join('demo_code/checkpoints/', args.model_name)
    args.ckp = os.path.join(args.model_save_dir,'best_model.pth')


    args.result_dir = os.path.join('demo_code/results/', args.model_name, 'infer_image')
    args.result_dir_rgb = os.path.join('demo_code/results/', args.model_name, 'infer_image_rgb')
    main(args)

