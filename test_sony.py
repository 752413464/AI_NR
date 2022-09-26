
from email.policy import strict
from operator import index
import torch
import os
import numpy as np
import skimage.metrics
import time
from torchvision.transforms import functional as F
import skimage
# from skimage.metrics import peak_signal_noise_ratio
from dataloader.data_loader_sony import test_dataloader
from dataloader.data_process_sony import inv_k_reno, write_back_tiff, inv_normalization,write_image, inv_k


def _test_sony(model, args):

    print(args.ckp, args.criterion, args.Iso)
    state_dict = torch.load(args.ckp)

    # model.load_state_dict(state_dict['model'])

    model.load_state_dict(state_dict, strict = False)

    dataloader = test_dataloader(batch_size=1, num_workers=0, camera = args.test_camera, iso = args.Iso)

    torch.cuda.empty_cache()
  
    black_level = 512
    white_level = 16383

    cost_time = []
    
    psnr_list = []
    ssim_list = []

    psnr_ori_list= []
    ssim_ori_list =[]

    # Height = 356
    # Width = 532

    Height = 712
    Width = 1064

    iso = args.Iso
    
    with torch.no_grad():

        model.eval()

        print('-------------------------test stage-------------------------')
        for idx, data in enumerate(dataloader):

                ep = time.time()

                input_img, label_img , H, W, name = data
        
                input_img = input_img.squeeze(dim=1)
                input_img = input_img.to(args.device)
                label_img = label_img.squeeze(dim=1)
      


        #     i_list = []
        #     for i in range(int(H / Height)):
        #         j_list = []
        #         for j in range(int(W / Width)):
        #             input_patch = input_img[:, :, i * Height:i * Height + Height, j * Width:j * Width + Width]
        #             j_list.append(input_patch)
        #         i_list.append(j_list)
        #     ii_list = []
        #     for i in i_list:
        #         jj_list = []
        #         for j in i:
        #             pred = model(j)
        #             jj_list.append(pred)
       
        #         ii_list.append(jj_list)
        #     pred = torch.zeros(input_img.size())
            

        #     for i_index, h in enumerate(ii_list):
        #         for j_index, w in enumerate(h):
        #             pred[:, :, i_index * Height:i_index * Height + Height, j_index * Width:j_index * Width + Width] = w


                pred = model(input_img)

                elapsed = time.time() - ep

                cost_time.append(elapsed)

                result_data = pred.cpu().detach().numpy().transpose(0, 2, 3, 1)
                
                height = result_data.shape[1]
                width = result_data.shape[2]

                result_data_k = inv_k(result_data, iso, args.test_camera)

                result_data_inv = inv_normalization(result_data_k, black_level, white_level)
                
                result_write_data = write_image(result_data_inv, height, width)

                
                input = input_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
                
                input_k = inv_k(input, iso, args.test_camera)
                
                input_inv = inv_normalization(input_k, black_level, white_level)
                
                input_write_data = write_image(input_inv, height, width)

                
                gt = label_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
                
                
                gt_inv = inv_normalization(gt, black_level, white_level)
                
                gt_write_data = write_image(gt_inv, height, width)


                psnr = skimage.metrics.peak_signal_noise_ratio(
                        gt_write_data.astype(np.float32), result_write_data.astype(np.float32), data_range=65535)
                ssim = skimage.metrics.structural_similarity(
                        gt_write_data.astype(np.float32), result_write_data.astype(np.float32),  data_range=65535)
                        
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                orig_psnr = skimage.metrics.peak_signal_noise_ratio(
                        gt_write_data.astype(np.float32), input_write_data.astype(np.float32), data_range=65535)
                orig_ssim = skimage.metrics.structural_similarity(
                        gt_write_data.astype(np.float32),  input_write_data.astype(np.float32), data_range=65535)
                        
                change_psnr = psnr - orig_psnr
                change_ssim = ssim - orig_ssim

                psnr_ori_list.append(orig_psnr)
                ssim_ori_list.append(orig_ssim)

                print('model : %s , dataset: %s \n'%(args.model_name ,args.dataset))
                print('scene:%d' %idx , 'psnr:-----%.4f' %psnr , '-----ssim-----%.4f'%ssim, '-----%.4f' %(orig_psnr), '---%.4f---'%(orig_ssim) , '[%.4f]'%(change_psnr), '[%.4f] \n'%(change_ssim))
                
                save_name = name[0]

                result_output_dir = str(args.result_dir) + '/' + str(args.model_name) + '_' + str(save_name.split('_')[0]) + '_ISO' + str(args.Iso) + '_' + args.test_camera + '.dng'

                input_dir = str(args.result_dir) + '/'  + 'noisy_' + str(save_name.split('_')[0]) + '_ISO' + str(args.Iso)  + '_' + args.test_camera + '.dng'
                
                label_dir = str(args.result_dir) + '/' + 'gt_' + str(save_name.split('_')[0]) + '.dng'
                
                write_back_tiff(input_dir, input_k, H, W , black_level, white_level )

                write_back_tiff(result_output_dir, result_data_k, H, W , black_level, white_level )
        
                write_back_tiff(label_dir, gt_k, H, W , black_level, white_level )

        ave_psnr = sum(psnr_list) / len(psnr_list)
        ave_ssim = sum(ssim_list) / len(ssim_list)

        ave_orig_psnr = sum( psnr_ori_list) / len( psnr_ori_list)
        ave_orig_ssim = sum( ssim_ori_list ) / len( ssim_ori_list)

        change_ave_psnr = ave_psnr - ave_orig_psnr
        change_ave_ssim = ave_ssim - ave_orig_ssim

        [w, psnr_max, psnr_min, ssim_min] =  [0.8, 60, 30, 0.8]

        score = (w * max(ave_psnr - psnr_min, 0) / (psnr_max - psnr_min) + (1 - w) * max(ave_ssim - ssim_min, 0) / (1 - ssim_min)) * 100


        print('----Score:%.4f----PSNR %.4f---SSIM %.4f----orig_psnr --- %.4f ---orig_ssim %.4f ---- , [%.4f] , [%.4f]' % (score, ave_psnr,ave_ssim , ave_orig_psnr,  ave_orig_ssim ,change_ave_psnr, change_ave_ssim))

        ave_cost = sum(cost_time) / len(cost_time)

        print("Average time: %f" % ave_cost)

 

    
