from cProfile import label
import os
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
import skimage
from dataloader.data_process_sony import inv_k_tensor, inv_normalization, write_image, inv_k
from dataloader.data_loader_sony import train_dataloader, valid_dataloader
from losses import losses
from tqdm import tqdm
import matplotlib.pyplot as plt


def _train_sony(model, args):


    torch.manual_seed(521)

    criterion_list = {
                    'l1': torch.nn.L1Loss('mean').to(args.device),
                    'l1_aug': torch.nn.L1Loss('mean').to(args.device),
                    'l1_none':torch.nn.L1Loss('mean').to(args.device),
                    'l2': torch.nn.MSELoss(reduction='mean').to(args.device)  
    }

    criterion = criterion_list[args.criterion]

    optimizer = optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=args.weight_decay,
                                 )

     ######### Scheduler-warmup+cosine ###########
    if args.warm_up:

        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-6)
        
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_cosine)
    else:
           scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-6)
   
    train_loader = train_dataloader(args.batch_size, args.num_worker, args.train_camera, args.device)
    max_iter = len(train_loader)
    valida_loader = valid_dataloader(args.batch_size, args.num_worker, args.train_camera)

    print(args.ckp, args.num_epoch, args.device, args.criterion)

    try:
           
            state = torch.load(args.ckp)
            # epoch = state['epoch']
            # optimizer.load_state_dict(state['optimizer'])
            # scheduler.load_state_dict(state['scheduler'])
            # model.load_state_dict(state['model'])
            model.load_state_dict(state, strict=False)
            print('--------------------------train stage-------------------------')
            
    except:
           
            print('-----------------------------ckp faild-----------------------------')
            

  #sony

    # Height = 128
    # Width = 128

    Height = 512
    Width = 512

    # Height = 356
    # Width = 532

    black_level = 512
    white_level = 16383
    best_loss = 0.1
    epoch_loss = []
    Score = []

    for epoch_idx in range(args.num_epoch):
        loss_list = []
        model.train()
        for iter_idx, batch_data in tqdm(enumerate(train_loader)):

            input_img, label_img, H, W, iso = batch_data

            input_img = input_img.to(args.device)
            label_img = label_img.to(args.device)

            input_img = input_img.squeeze(dim=1)
            label_img = label_img.squeeze(dim=1)

            for i in range(int(H / Height)):  
                 for j in range(int(W/ Width)):

                    input_patch = input_img[:,:,i * Height:i * Height + Height, j * Width:j * Width + Width]
                    label_patch = label_img[:,:,i * Height:i * Height + Height, j * Width:j * Width + Width]
    
                    pred_patch = model(input_patch)

                    # print('pred_patch =', pred_patch.max())

                    # pred_patch = inv_k_tensor(pred_patch, iso, args.device, camera = args.train_camera)

                    # label_patch= inv_k_tensor(label_patch, iso, args.device, camera = args.train_camera)
                    # for k in range(len(pred_patch)):

                    #     loss = 0.
                    #     loss += criterion(torch.clamp(pred_patch[k], 0, 1), label_patch)

                    # total_loss = loss / len(pred_patch)
                    if args.criterion == 'l1_aug':
                        total_loss = criterion(pred_patch, label_patch)
                        loss = total_loss   

                    elif args.criterion == 'l1':
                        total_loss = criterion(pred_patch, label_patch)
                        loss = total_loss     

                    elif args.criterion == 'l1_none':
                       
                        total_loss = criterion(pred_patch, label_patch)
                        loss = total_loss  
      
                    optimizer.zero_grad()
                    loss.backward()
                    loss_list.append(loss.item())
                    optimizer.step()

            print(" Epoch: %03d/%03d Iter:%4d/%4d lr: %.6f Loss: %7.6f  \n" % (epoch_idx, args.num_epoch, iter_idx + 1, max_iter, optimizer.param_groups[0]['lr'], loss.item()))
        
        ave_loss = sum(loss_list) / len(loss_list)
        epoch_loss.append(ave_loss)
        plt.subplot(121)
        plt.plot(epoch_loss, linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('training loss curve')
        scheduler.step()
   
        if best_loss >= ave_loss:
            torch.save(model.state_dict(), os.path.join(args.model_save_dir, args.dataset + args.criterion + args.train_camera + '.pth'))
            print('----best_loss = %.6f'%best_loss , 'last_loss = %.6f----' %ave_loss)
            best_loss = ave_loss
        else:
            print('----best_loss = %.6f'%best_loss , 'last_loss = %.6f----' %ave_loss)
            print('.........................Converged...................')

        psnr_list = []
        ssim_list = []
        psnr_ori_list= []
        ssim_ori_list =[]


        with torch.no_grad():

            model.eval()

            print('--------------------------validation stage---------------------')
            for idx, data in enumerate(valida_loader):

                input_img, label_img, _, _, iso = data

                input_img = input_img.squeeze(dim=1)
                label_img = label_img.squeeze(dim=1)

    
                input_img = input_img.to(args.device)

          
                # i_list = []
                # for i in range(int(H / Height)):
                #     j_list = []
                #     for j in range(int(W / Width)):
                #         input_patch = input_img[:, :, i * Height:i * Height + Height, j * Width:j * Width + Width]
                #         j_list.append(input_patch)
                #     i_list.append(j_list)
                # ii_list = []
                # for i in i_list:
                #     jj_list = []
                #     for j in i:
                #         pred = model(j)
                #         jj_list.append(pred)

                #     ii_list.append(jj_list)
                # pred = torch.zeros(input_img.size())
                
                # for i_index, h in enumerate(ii_list):
                #     for j_index, w in enumerate(h):
                #         pred[:, :, i_index * Height:i_index * Height + Height, j_index * Width:j_index * Width + Width] = w

                pred = model(input_img)

                result_data = pred.cpu().detach().numpy().transpose(0, 2, 3, 1)
         
                height = result_data.shape[1]
                width = result_data.shape[2]

                result_data_k = inv_k(result_data, iso, args.train_camera)

                result_data_inv = inv_normalization(result_data_k, black_level, white_level)
               
                result_write_data = write_image(result_data_inv, height, width)

               
                input = input_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
                
                input_k = inv_k(input, iso, args.train_camera)
                
                input_inv = inv_normalization(input_k, black_level, white_level)
               
                input_write_data = write_image(input_inv, height, width)

                gt = label_img.cpu().detach().numpy().transpose(0, 2, 3, 1)


                gt_inv = inv_normalization(gt, black_level, white_level)
        
                gt_write_data = write_image(gt_inv, height, width)

                psnr = skimage.metrics.peak_signal_noise_ratio(
                    gt_write_data.astype(np.float32), result_write_data.astype(np.float), data_range= 65535)
                ssim = skimage.metrics.structural_similarity(
                    gt_write_data.astype(np.float32), result_write_data.astype(np.float),  data_range= 65535)
                
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                orig_psnr = skimage.metrics.peak_signal_noise_ratio(
                    gt_write_data.astype(np.float), input_write_data.astype(np.float), data_range= 65535)
                orig_ssim = skimage.metrics.structural_similarity(
                    gt_write_data.astype(np.float),  input_write_data.astype(np.float), data_range= 65535)
                
                change_psnr = psnr - orig_psnr
                change_ssim = ssim - orig_ssim

        
                psnr_ori_list.append(orig_psnr)
                ssim_ori_list.append(orig_ssim)

                print('model : %s , dataset: %s , camera: %s\n'%(args.model_name ,args.dataset, args.train_camera))
                print('scene:%d' %idx , 'psnr:-----%.4f' %psnr , '-----ssim-----%.4f'%ssim, '-----%.4f' %(orig_psnr), '---%.4f---'%(orig_ssim) , '[%.4f]'%(change_psnr), '[%.4f] \n'%(change_ssim))
 
            ave_psnr = sum(psnr_list) / len(psnr_list)
            ave_ssim = sum(ssim_list) / len(ssim_list)

            ave_orig_psnr = sum( psnr_ori_list) / len( psnr_ori_list)
            ave_orig_ssim = sum( ssim_ori_list ) / len( ssim_ori_list)


            change_ave_psnr = ave_psnr - ave_orig_psnr
            change_ave_ssim = ave_ssim - ave_orig_ssim

            [w, psnr_max, psnr_min, ssim_min] =  [0.8, 60, 30, 0.8]
            
            score = (w * max(ave_psnr - psnr_min, 0) / (psnr_max - psnr_min) + (1 - w) * max(ave_ssim - ssim_min, 0) / (1 - ssim_min)) * 100
            Score.append(score)
            
            plt.subplot(122)
            plt.plot(Score, linewidth=1)
            plt.xlabel('Epoch')
            plt.ylabel('%')
            plt.title('Score')
            plt.savefig(os.path.join(args.model_save_dir, args.dataset + args.criterion + args.train_camera + 'loss_curve.png'))
            print('Epoch%03d ----Score:%.4f----PSNR %.4f---SSIM %.4f----orig_psnr --- %.4f ---orig_ssim %.4f ---- , [%.4f] , [%.4f]' % (epoch_idx, score, ave_psnr,ave_ssim , ave_orig_psnr,  ave_orig_ssim ,change_ave_psnr, change_ave_ssim))
      
    save_name = os.path.join(args.model_save_dir, args.dataset + args.criterion + args.train_camera +'_final.pth')
    torch.save(model.state_dict(), save_name)


