import os
import torch
import argparse
from torch.backends import cudnn


from model.MPRNet import MPRNet
from model.PMRID import PMRID
from model.PMRID_caplusp import PMRID_caplusp
from model.cycleisp import isp
from model.PMRID_up import PMRID_up

from train_zte import _train_zte
from test_zte import _test_zte

from train_sony import _train_sony
from test_sony import _test_sony

from train_reno import _train_reno
from test_reno import _test_reno

from train_sidd import _train_sidd
from test_sidd import _test_sidd


def main(args):

    net_list = {
        'PMRID': PMRID(),
        'PMRID_up': PMRID_up(),
        'MPR': MPRNet(),
        'PMRID_caplusp':PMRID_caplusp(),
        'isp':isp()
    }
 
    net = net_list[args.model_name]

    model = net.to(args.device)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)


    torch.cuda.empty_cache()

    if args.dataset == 'zte':

        if not os.path.exists(args.result_dir):
                    os.makedirs(args.result_dir)  

        if args.mode == 'train':
            _train_zte(model, args) 
        else:
            _test_zte(model, args)

    elif args.dataset =='sony':
        
        if not os.path.exists(args.result_dir):
                    os.makedirs(args.result_dir)

        if args.mode == 'train':
            _train_sony(model, args) 
        else:
            _test_sony(model, args)
          
    elif args.dataset =='reno':
        
        if not os.path.exists(args.result_dir):
                    os.makedirs(args.result_dir)

        if args.mode == 'train':
            _train_reno(model, args) 
        else:
            _test_reno(model, args)
              
    elif args.dataset =='sidd':
        
        if not os.path.exists(args.result_dir):
                    os.makedirs(args.result_dir)

        if args.mode == 'train':
            _train_sidd(model, args) 
        else:
            _test_sidd(model, args)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('--model_name', default='PMRID_caplusp',  type=str)
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--dataset', default='sony', choices=['sony' , 'zte' , 'reno'], type = str)

    # Train
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4) # default=1e-4
    parser.add_argument('--weight_decay', type=float, default=0) ## default=0, 5e-2
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--device', type=str, default = 'cuda:5')
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--test_dir', type=str, default='data/valid/')
    parser.add_argument('--criterion', type = str, default = 'l1_none')
    parser.add_argument('--warm_up', type = bool, default = False)

    #Test
    parser.add_argument('--Iso', type = int, default = '3200')
    parser.add_argument('--train_camera', type = str, default = 'ip')
    parser.add_argument('--test_camera', type = str, default = 's6')
    args = parser.parse_args()

    args.model_save_dir = os.path.join('demo_code/checkpoints/', args.model_name)
    args.ckp = os.path.join(args.model_save_dir, args.dataset + args.criterion + args.train_camera + '.pth')
    # args.ckp = ('/mnt/AI denoise/weights/MPRNet_best_64_L1_aug.pth')
    args.result_dir = os.path.join('demo_code/results/', args.model_name, str(args.dataset) + '_image')

    main(args)

