import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
import skimage
from skimage.metrics import peak_signal_noise_ratio
from dataloader.data_process import inv_normalization,write_image,read_image,write_back_dng
from dataloader.data_loader import train_dataloader,valid_dataloader
from losses import losses
from tqdm import tqdm
import numpy as np



# def accumulate(model1, model2, decay=0.999):
#     par1 = dict(model1.named_parameters())
#     par2 = dict(model2.named_parameters())

#     for k in par1.keys():
#         par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def _train(model, args):

    # criterion = torch.nn.L1Loss()
    criterion = losses.CharbonnierLoss()
    # mse_criterion =  torch.nn.MSELoss(reduce=True, size_average=True)
    # l1_criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=args.weight_decay,
                                 )
    ######### Scheduler-warmup+cosine ###########
    warmup_epochs = 3
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=4, min_lr=1e-6,
    #                                                        eps=1e-8)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch-warmup_epochs+40, eta_min=1e-6)
    # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    
    ######### Scheduler-MultiStepLR ###########
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)

    train_loader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(train_loader)
    valida_loader = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    # if args.ckp:
    print({args.ckp})
    try:
            state = torch.load(args.ckp)
            # epoch = state['epoch']
            # optimizer.load_state_dict(state['optimizer'])
            # scheduler.load_state_dict(state['scheduler'])
            model.load_state_dict(state['model'])
            print('------------------------------train stage------------------------------')
            
    except:
            print('--------------------------------ckp faild--------------------------------')
    
    # writer = SummaryWriter()

    best_psnr = -1
    best_ssim = -1
   
    model.train()
    for epoch_idx in range(args.num_epoch):
    
        for iter_idx, batch_data in tqdm(enumerate(train_loader)):
            input_img, label_img = batch_data
            input_img = input_img.to(args.device)
            label_img = label_img.to(args.device)
            input_img = input_img.squeeze(dim=1)
            label_img = label_img.squeeze(dim=1)
            optimizer.zero_grad()
            pred_img = model(input_img)
            # label_fft = torch.fft.fft2(label_patch, dim=(-2, -1))
            # pred_fft = torch.fft.fft2(pred_img[0], dim=(-2, -1)) 
            # loss_fft = l1_criterion(pred_fft, label_fft)
            # loss = 0.9 * loss_content + 0.1 * loss_fft

            loss= criterion(torch.clamp(pred_img,0,1),label_img)
            loss.backward()
            optimizer.step()
           
            # accumulate(model_running, model)

            # iter_pixel_adder(loss.item())
            #         # iter_fft_adder(loss_fft.item())
            # epoch_pixel_adder(loss.item())
            #        # epoch_fft_adder(loss_fft.item())
                    
            # if (iter_idx + 1) % args.print_freq == 0:
            # lr = check_lr(optimizer)
            
            print(" Epoch: %03d Iter:[%4d/%4d] lr: %.5f Loss: %7.4f  \n" % (epoch_idx, iter_idx + 1, max_iter, optimizer.param_groups[0]['lr'], loss.item()))
            # writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
            # writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
            # iter_timer.tic()
            # iter_pixel_adder.reset()
                # iter_fft_adder.reset()
        # if epoch_idx > 100 and  epoch_idx % args.save_freq == 0:
        #     save_name = os.path.join(args.model_save_dir, 'model_%d.pth' % epoch_idx)
        #     torch.save({'model': model.state_dict()}, save_name)
            # torch.save({'model': model_running.state_dict()}, os.path.join(args.model_save_dir, str(epoch_idx).zfill(5)+'running'+'.pth'))

        # print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f " % (
        #     epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average()))
        # # epoch_fft_adder.reset()
        # epoch_pixel_adder.reset()
        scheduler_cosine.step()

        # scheduler.step()
      

        model.eval()
        psnr_list = []
        ssim_list = []

        black_level = 1024
        white_level = 16383

        with torch.no_grad():

            print('------------------------------validation stage-------------------------')
            for idx, data in enumerate(valida_loader):
                input_img, label_img , name = data
                input_img = input_img.squeeze(dim=1)
                label_img = label_img.squeeze(dim=1)
                input_img = input_img.to(args.device)
        
                pred = model(input_img)
                
                result_data = pred.cpu().detach().numpy().transpose(0, 2, 3, 1)
                height = result_data.shape[1]
                width = result_data.shape[2]
                # result_data = result_data.reshape(-1, height // 2, width // 2, 4)
                result_data = inv_normalization(result_data, black_level, white_level)
                result_write_data = write_image(result_data, height, width)

                input = input_img.cpu().detach().numpy().transpose(0, 2, 3, 1)

                input = inv_normalization(input, black_level, white_level)
                input_write_data = write_image(input, height, width)

                gt = label_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
                # gt = gt.reshape(-1, height // 2, width // 2, 4)
                gt = inv_normalization(gt, black_level, white_level)
                gt = write_image(gt, height, width)

                psnr = skimage.metrics.peak_signal_noise_ratio(
                    gt.astype(np.float), result_write_data.astype(np.float), data_range=white_level)
                ssim = skimage.metrics.structural_similarity(
                    gt.astype(np.float), result_write_data.astype(np.float), multichannel=True, data_range=white_level)
                
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                orig_psnr = skimage.metrics.peak_signal_noise_ratio(
                    gt.astype(np.float), input_write_data.astype(np.float), data_range=white_level)
                orig_ssim = skimage.metrics.structural_similarity(
                    gt.astype(np.float),  input_write_data.astype(np.float), data_range=white_level)
                print('scene:%d' %idx , 'psnr:-----%.4f' %psnr , '-----ssim-----%.3f'%ssim, '-----[%.4f]---' %(psnr-orig_psnr), '[%.3f] \n'%(ssim-orig_ssim)  )
 
            ave_psnr = sum(psnr_list) / len(psnr_list)
            ave_ssim = sum(ssim_list) / len(ssim_list)

            print('Epoch%03d --------- Average  PSNR %.4f dB , SSIM --------%.3f' % (epoch_idx, ave_psnr,ave_ssim))

            # writer.add_scalar('PSNR_zet', psnr, epoch_idx)
            # writer.add_scalar('SSIM_zet', ssim, epoch_idx)

            if ave_psnr >= best_psnr and ave_ssim >= best_ssim:
                best_psnr = psnr
                best_ssim = ssim
               
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'best_model.pth'))
    save_name = os.path.join(args.model_save_dir, 'final.pth')
    torch.save({'model': model.state_dict()}, save_name)


