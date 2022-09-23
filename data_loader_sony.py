import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from dataloader.data_process_sony import normalization, unpack_image, add_noise, pg_noise, KSigma
from dataloader.bay_aug import bayer_aug
import glob
import rawpy



# def noise_profiles(camera):
#     camera = camera.lower()
#     if camera == 'ip':  # iPhone
#         iso_set = [100, 200, 400, 800, 1600, 2000]
#         cshot = [0.00093595, 0.00104404, 0.00116461, 0.00129911, 0.00144915, 0.00150104]
#         cread = [4.697713410870357e-07, 6.904488905478659e-07, 6.739473744228789e-07,
#                  6.776787431555864e-07, 6.781983208034481e-07, 6.783184262356993e-07]
#     elif camera == 's6':  # Sumsung s6 edge
#         iso_set = [100, 200, 400, 800, 1600, 3200]
#         cshot = [0.00162521, 0.00256175, 0.00403799, 0.00636492, 0.01003277, 0.01581424]
#         cread = [1.1792188420255036e-06, 1.607602896683437e-06, 2.9872611575167216e-06,
#                  5.19157563906707e-06, 1.0011034196248119e-05, 2.0652668477786836e-05]
#     elif camera == 'gp':  # Google Pixel
#         iso_set = [100, 200, 400, 800, 1600, 3200, 6400]
#         cshot = [0.00024718, 0.00048489, 0.00095121, 0.001866, 0.00366055, 0.00718092, 0.01408686]
#         cread = [1.6819349659429324e-06, 2.0556981890860545e-06, 2.703070976302046e-06,
#                  4.116405515789963e-06, 7.569256436438246e-06, 1.5199001098203388e-05, 5.331422827048082e-05]
#     elif camera == 'sony':  # Sony a7s2
#         iso_set = [800, 1600, 3200]
#         cshot = [1.0028880020069384, 1.804521362114003, 3.246920234173119]
#         cread = [4.053034401667052, 6.692229120425673, 4.283115294604881]
#     elif camera == 'nikon':  # Nikon D850
#         iso_set = [800, 1600, 3200]
#         cshot = [3.355988883536526, 6.688199969242411, 13.32901281288985]
#         cread = [4.4959735547955635, 8.360429952584846, 15.684213053647735]
#     else:
#         assert NotImplementedError
#     return iso_set, cshot, cread



class DngTrainset(Dataset):

    def __init__(self, camera, device):

        train_ids = glob.glob('Sony/long/' + '*.ARW')

        self.train_ids = train_ids[:215]


        self.camera = camera

        self.device = device

        if self.camera == 'gp':

            self.k_coeff = [2.19586075e-06, 8.43112069e-05]

            self.b_coeff = [1.11039570e-12, 8.57579026e-10, 2.15851893e-06]

        elif self.camera == 's6':

            self.k_coeff = [4.48916546e-06, 2.02585627e-03]

            self.b_coeff = [2.27373832e-13, 5.53331696e-09, 6.10968591e-07]

        elif self.camera == 'ip':

             self.k_coeff = [2.74289431e-07, 9.99170650e-04]

             self.b_coeff = [ 3.97667623e-21, -1.98173628e-17, 3.46822012e-14, -2.55887922e-11,
                        7.74059183e-09, -8.11401512e-08]

        elif self.camera == 'sy':
        
             self.k_coeff =  [5.86116626e-08,1.77486337e-05]

             self.b_coeff = [-1.15787697e-21, -2.53544295e-18, -1.28930515e-15, 2.97640275e-11, 1.76379864e-07, 9.72987856e-05]
       
        elif self.camera == 're':

             self.k_coeff  = [2.35108510e-06, 3.40729804e-05]
 
             self.b_coeff = [1.08609445e-11, 9.94115206e-09, 1.75367559e-06]

        self.k_sigma = KSigma(
            k_coeff = self.k_coeff,
            b_coeff = self.b_coeff,
            anchor = 1600,
        )

        
        # self.train_ids = []
        # for img in train_fns:
        #     self.train_ids.append(os.path.basename(img))
        # self.train_ids.sort()
   
    def __len__(self):

        return len(self.train_ids)


    def __getitem__(self, idx):
    
        black_level = 512

        white_level = 16383

        iso_set = [100, 200, 400, 800, 1600, 3200, 6400]

        # iso_set_pro = [1600, 3200, 6400]

        # cshot = [0.00024718, 0.00048489, 0.00095121, 0.001866, 0.00366055, 0.00718092, 0.01408686]

        # cread = [1.6819349659429324e-06, 2.0556981890860545e-06, 2.703070976302046e-06,
        #     4.116405515789963e-06, 7.569256436438246e-06, 1.5199001098203388e-05, 5.331422827048082e-05]
   
        iso = iso_set[np.random.choice(len(iso_set))]
        
        # iso = iso_set_pro[np.random.choice(len(iso_set_pro))]

        # k, read_scale, iso =  cshot[i], cread[i], iso_set[i]

        k = np.poly1d(self.k_coeff)(iso)

        read_scale = np.poly1d(self.b_coeff)(iso)

        gt_dir = self.train_ids[idx]

        # gt_name = os.path.basename(gt_path)

        gt_raw = rawpy.imread(gt_dir)
        
        label = gt_raw.raw_image_visible

        gt_raw.close()
        
        label_norm = normalization(label, black_level, white_level)

        # label_norm = bayer_aug(label_norm, 'RGGB')

        label_expand= np.expand_dims(label_norm, 2).transpose(2, 0, 1)

        # label_expand = np.ascontiguousarray(label_expand)
     
        label_tensor = torch.from_numpy(label_expand).float()

  
        input_tensor = pg_noise(label_tensor, k, read_scale)

        input_tensor = np.clip(input_tensor, 0, 1)



        input_tensor_k = self.k_sigma(input_tensor, iso)

        label_tensor_k = self.k_sigma(label_tensor, iso)

        # print('input k max = ', input_tensor_k.max())

        # label_tensor_k = self.k_sigma(label_tensor, iso)

        # print('label k max = ', label_tensor_k.max())

        input_tensor_k = unpack_image(input_tensor_k)

        label_tensor_k = unpack_image(label_tensor_k)

        # print('label_tensor =', label_tensor.max())

        # print('input_tensor = ', input_tensor_k.max())

        # print('iso = ', iso)

        H , W = label_tensor_k.shape[2:4]

        ##add noise
        # noise = random.random() < 0.5
        # if noise:        
        #     # shot_noise, read_noise = random_noise_levels()
        #     # image  = add_noise(image, shot_noise, read_noise)
        #     image = add_noise(image)

        return input_tensor_k, label_tensor_k, H, W, iso


class DngValidset(Dataset):

    def __init__(self, camera):

        valid_fns = glob.glob('Sony/long/' + '2*.ARW')

        self.valid_ids = valid_fns[5:10:2]

        self.camera = camera

        if self.camera == 'gp':

            self.k_coeff = [2.19586075e-06, 8.43112069e-05]

            self.b_coeff = [1.11039570e-12, 8.57579026e-10, 2.15851893e-06]

        elif self.camera == 's6':

            self.k_coeff = [4.48916546e-06, 2.02585627e-03]

            self.b_coeff = [2.27373832e-13, 5.53331696e-09, 6.10968591e-07]

        
        elif self.camera == 'ip':

             self.k_coeff = [2.74289431e-07, 9.99170650e-04]
             
             self.b_coeff = [ 3.97667623e-21, -1.98173628e-17, 3.46822012e-14, -2.55887922e-11,
                        7.74059183e-09, -8.11401512e-08]

        elif self.camera == 'sy':
        
             self.k_coeff =  [5.86116626e-08,1.77486337e-05]

             self.b_coeff = [-1.15787697e-21, -2.53544295e-18, -1.28930515e-15, 2.97640275e-11, 1.76379864e-07, 9.72987856e-05]
        
        elif self.camera == 're':
            
             self.k_coeff  = [2.35108510e-06, 3.40729804e-05]
 
             self.b_coeff = [1.08609445e-11, 9.94115206e-09, 1.75367559e-06]

    def __len__(self):

        return len(self.valid_ids)


    def __getitem__(self, idx):

        black_level = 512

        iso = 3200

        white_level = 16383


        self.k_sigma = KSigma(
            k_coeff = self.k_coeff,
            b_coeff = self.b_coeff,
            anchor = 1600,
        )


        k = np.poly1d(self.k_coeff)(iso)

        read_scale = np.poly1d(self.b_coeff)(iso)

        gt_dir = self.valid_ids[idx]

        # gt_name = os.path.basename(gt_path)

        gt_raw = rawpy.imread(gt_dir)
        
        label = gt_raw.raw_image_visible

        label = normalization(label, black_level, white_level)

        label= np.expand_dims(label, 2).transpose(2, 0, 1)
     
        label_tensor = torch.from_numpy(label).float()

        input_tensor = pg_noise(label_tensor, k, read_scale)

        input_tensor = np.clip(input_tensor, 0, 1)

        input_tensor_k = self.k_sigma(input_tensor, iso)

        input_tensor_k = unpack_image(input_tensor_k)

        label_tensor = unpack_image(label_tensor)

        H , W = label_tensor.shape[2:4]

        return input_tensor_k, label_tensor, H, W, iso




class DngTestset(Dataset):

    def __init__(self, camera, iso):

        test_fns = glob.glob('Sony/long/' + '1*.ARW')

        self.test_ids = test_fns[:6]

        self.camera = camera

        self.iso = iso

        # self.test_ids = []
        # for img in test_fns:
        #     self.test_ids.append(os.path.basename(img)[0:5])

        # self.test_ids = self.test_ids[:6]
        # print('Test examples = ',  len(self.test_ids))

    def __len__(self):
        return len(self.test_ids)
    
    def __getitem__(self, idx):

        black_level = 512

        white_level = 16383

        if self.camera == 'gp':

            self.k_coeff = [2.19586075e-06, 8.43112069e-05]

            self.b_coeff = [1.11039570e-12, 8.57579026e-10, 2.15851893e-06]

        elif self.camera == 's6':
    
            self.k_coeff = [4.48916546e-06, 2.02585627e-03]

            self.b_coeff = [2.27373832e-13, 5.53331696e-09, 6.10968591e-07]

        
        elif self.camera == 'ip':

             self.k_coeff = [2.74289431e-07, 9.99170650e-04]
             
             self.b_coeff = [ 3.97667623e-21, -1.98173628e-17, 3.46822012e-14, -2.55887922e-11,
                        7.74059183e-09, -8.11401512e-08]

        elif self.camera == 'sy':
        
             self.k_coeff =  [5.86116626e-08,1.77486337e-05]

             self.b_coeff = [-1.15787697e-21, -2.53544295e-18, -1.28930515e-15, 2.97640275e-11, 1.76379864e-07, 9.72987856e-05]
        
        elif self.camera == 're':
            
             self.k_coeff  = [2.35108510e-06, 3.40729804e-05]
 
             self.b_coeff = [1.08609445e-11, 9.94115206e-09, 1.75367559e-06]


        self.k_sigma = KSigma(
            k_coeff = self.k_coeff,
            b_coeff = self.b_coeff,
            anchor = 1600,
        )

        # iso_set = [100, 200, 400, 800, 1600, 3200, 6400]

        # cshot = [0.00024718, 0.00048489, 0.00095121, 0.001866, 0.00366055, 0.00718092, 0.01408686]

        # cread = [1.6819349659429324e-06, 2.0556981890860545e-06, 2.703070976302046e-06,
        #     4.116405515789963e-06, 7.569256436438246e-06, 1.5199001098203388e-05, 5.331422827048082e-05]
   
        iso = self.iso

        # k, read_scale, iso =  cshot[i], cread[i], iso_set[i]

        k = np.poly1d(self.k_coeff)(iso)

        read_scale = np.poly1d(self.b_coeff)(iso)

        gt_dir = self.test_ids[idx]

        gt_name = os.path.basename(gt_dir)

        gt_raw = rawpy.imread(gt_dir)
        
        label = gt_raw.raw_image_visible

        label = normalization(label, black_level, white_level)

        label= np.expand_dims(label, 2).transpose(2, 0, 1)
     
        label_tensor = torch.from_numpy(label).float()

        input_tensor = pg_noise(label_tensor, k, read_scale)

        input_tensor = np.clip(input_tensor, 0, 1)

        input_tensor_k = self.k_sigma(input_tensor, iso)

        # label_tensor_k = self.k_sigma(label_tensor, iso)

        input_tensor_k = unpack_image(input_tensor_k)

        label_tensor_k = unpack_image(label_tensor)

        H , W = label_tensor_k.shape[2:4]

       
      
      
        return input_tensor_k, label_tensor_k, H, W,  gt_name



def train_dataloader(batch_size, num_workers, camera, device):

    dataloader = DataLoader(
        DngTrainset(camera, device),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True
    )
    return dataloader



def valid_dataloader(batch_size = 1, num_workers = 0, camera = 'gp'):

    dataloader = DataLoader(
        DngValidset(camera),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader



def test_dataloader(batch_size=1, num_workers=0, camera = 'gp', iso = '6400'):
    dataloader = DataLoader(
        DngTestset(camera, iso), 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True
    )
    return dataloader
