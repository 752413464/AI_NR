from ast import Name
from operator import index
import torch
import os
import numpy as np
import skimage.metrics
import time
from torchvision.transforms import functional as F
from skimage.metrics import peak_signal_noise_ratio
from dataloader.data_loader import test_dataloader
from dataloader.data_process import inv_normalization,write_image,read_image,write_back_dng

def _test(model, args):
    print({args.ckp} )
    state_dict = torch.load(args.ckp)
    model.load_state_dict(state_dict['model'])
    # for m in model.modules():
    #     if hasattr(m, 'switch_to_deploy'):
    #         m.switch_to_deploy()

    # device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.test_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    model.eval()
    black_level = 1024
    white_level = 16383

    Height = 320    
    Width = 320
    cost_time = []
    
    # Height = 217
    # Width = 289

    with torch.no_grad():
        print('------------------------------test stage-----------------------------')
        for idx, data in enumerate(dataloader):
            ep = time.time()
            input_img, ph, pw, H, W, name = data
            input_img = input_img.squeeze(dim=1)
            input_img = input_img.to(args.device)
    
            i_list = []
            for i in range(int(H / Height)):
                j_list = []
                for j in range(int(W / Width)):
                    input_patch = input_img[:, :, i * Height:i * Height + Height, j * Width:j * Width + Width]
                    j_list.append(input_patch)
                i_list.append(j_list)
            ii_list = []
            for i in i_list:
                jj_list = []
                for j in i:
                    pred = model(j)
                    jj_list.append(pred)
                ii_list.append(jj_list)
            pred = torch.zeros(input_img.size())
            for i_index, h in enumerate(ii_list):
                for j_index, w in enumerate(h):
                    pred[:, :, i_index * Height:i_index * Height + Height, j_index * Width:j_index * Width + Width] = w
            
            pred = model(input_img)
            elapsed = time.time() - ep
            cost_time.append(elapsed)
            # image, height, width = read_image('../data/test/noisy0.dng')
            result_data = pred.cpu().detach().numpy().transpose(0, 2, 3, 1)
            # result_data = result_data.reshape(-1, height // 2, width // 2, 4)
            result_data_inor = inv_normalization(result_data, black_level, white_level)
            result = result_data_inor[:, ph:-ph, pw:-pw, :]

            result_write_data = write_image(result, H, W)

            result_output_dir = str(args.result_dir) + '/' + str(args.model_name) + str(idx) + 'p.dng'

            write_back_dng(name[0], result_output_dir, result_write_data)

            print('=======================%d iter  time: %f=====================' % (idx + 1, elapsed))

        ave_cost = sum(cost_time) / len(cost_time)
        print("Average time: %f" % ave_cost)

 

