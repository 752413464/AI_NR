import os
# from tkinter import image_types
import numpy as np
import rawpy
import torch
import skimage.metrics
import argparse
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
# from dataloader.data_augment import PairCompose, PairToTensor, PairRandomCrop, PairRandomHorizontalFilp, random_noise_levels, add_noise
from dataloader.data_process import normalization, read_image
import random
import glob



class DngTrainset(Dataset):

    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.input_list = os.listdir(os.path.join(image_dir, 'noisy'))
        self.ground_list = os.listdir(os.path.join(image_dir, 'ground_truth'))
        
        self._check_image(self.input_list)
        self.input_list.sort()
        self._check_image(self.ground_list)
        self.ground_list.sort()
       
    def __len__(self):

        return len(self.input_list)

    def __getitem__(self, idx):

        black_level = 1024
        white_level = 16383
        image, height, width = read_image(os.path.join(self.image_dir, 'noisy', self.input_list[idx]))
        image = normalization(image, black_level, white_level)
        label, height, width = read_image(os.path.join(self.image_dir, 'ground_truth', self.ground_list[idx]))
        label = normalization(label, black_level, white_level)

        patch_x = 512
        patch_y = 512

        xx = np.random.randint(0,(width/2) - patch_x)
        yy = np.random.randint(0,(height/2) -patch_y)

        image_patch = image[yy:yy+patch_y,xx:xx+patch_x,:]
        label_patch = label[yy:yy+patch_y,xx:xx+patch_x,:]

        # Data Augmentations
        # if np.random.randint(2,size=1)[0] == 1:  # random flip 
        #     input_patch = np.flip(input_patch, axis=0)
        #     gt_patch = np.flip(gt_patch, axis=1)
        # if np.random.randint(2,size=1)[0] == 1: 
        #     input_patch = np.flip(input_patch, axis=1)
        #     gt_patch = np.flip(gt_patch, axis=2)
  
        image_nchw= np.expand_dims(image_patch, axis = 0).transpose(0,3,1,2)
        image_tensor = torch.from_numpy(image_nchw).float()  

        label_nchw = np.expand_dims(label_patch, axis = 0).transpose(0,3,1,2)
        label_tensor = torch.from_numpy(label_nchw).float()


  
        ##add noise
        # noise = random.random() < 0.5
        # if noise:        
        #     # shot_noise, read_noise = random_noise_levels()
        #     # image  = add_noise(image, shot_noise, read_noise)
        #     image = add_noise(image)
  
    
        return image_tensor, label_tensor

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] == 'DS_Store':
                continue
            if splits[-1] not in ['dng']:
                raise ValueError


class DngValidset(Dataset):

    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.input_list = os.listdir(os.path.join(image_dir, 'noisy'))
        self.ground_list = os.listdir(os.path.join(image_dir, 'ground_truth'))
        
        self._check_image(self.input_list)
        self.input_list.sort()
        self._check_image(self.ground_list)
        self.ground_list.sort()
       
       
    def __len__(self):
        # print('valid examples = ------------' ,len(self.input_list))
        return len(self.input_list)

    def __getitem__(self, idx):

        black_level = 1024
        white_level = 16383
        image, height, width = read_image(os.path.join(self.image_dir, 'noisy', self.input_list[idx]))
        image = normalization(image, black_level, white_level)
        name = self.input_list[idx]

        label, height, width = read_image(os.path.join(self.image_dir, 'ground_truth', self.ground_list[idx]))
        label = normalization(label, black_level, white_level)

        # label = torch.from_numpy(np.transpose(label.reshape(-1, height//2, width//2, 4), (0, 3, 1, 2))).float()

        patch_x = 1024
        patch_y = 1024

        # xx = np.random.randint(0,(width/2) - patch_x)
        # yy = np.random.randint(0,(height/2) -patch_y)


        image_patch = image[0:patch_y,0:patch_x,:]
        label_patch = label[0:patch_y,0:patch_x,:]


        image_nchw= np.expand_dims(image_patch, axis = 0).transpose(0,3,1,2)
        image_tensor = torch.from_numpy(image_nchw).float()  

        label_nchw = np.expand_dims(label_patch, axis = 0).transpose(0,3,1,2)
        label_tensor = torch.from_numpy(label_nchw).float()
 
      
        return image_tensor, label_tensor,name

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] == 'DS_Store':
                continue
            if splits[-1] not in ['dng']:
                raise ValueError



# class DngTestset(Dataset):
#     def __init__(self, image_dir, transform=None):

#         # self.image_dir = os.path.join(image_dir,'test')
#         self.input_list = os.listdir(self.image_dir)
#         self.image_dir = glob.glob('data/valid/noisy/' + '*.dng')
        
#         self._check_image(self.input_list)
#         self.input_list.sort()
#         self.transform = transform


#     def __len__(self):
      
#         return len(self.input_list)

#     def __getitem__(self, idx):
#         black_level = 1024
#         white_level = 16383
#         image, height, width = read_image(os.path.join(self.image_dir,self.input_list[idx]))
#         image = normalization(image, black_level, white_level)
#         image = torch.from_numpy(np.transpose(image.reshape(-1, height//2, width//2, 4), (0, 3, 1, 2))).float()
#         name = self.input_list[idx]
#         return image,name



class DngTestset(Dataset):

    def __init__(self, test_dir):
        self.test_list = glob.glob( test_dir + '*')
        print('Test examples =  %d ------------' %len(self.test_list))
        # # self._check_image(self.test_list)
        # self.test_list.sort()
        # print('Test examples:', len(self.inference_list))
    
    def __getitem__(self, index):
        black_level = 1024
        white_level = 16383

        image_rggb, _, _  = read_image(self.test_list[index])
        height, width = image_rggb.shape[0:2]
        ph, pw = (32-(height % 32))//2, (32-(width % 32))//2
        image_pad = np.pad(image_rggb, [(ph, ph), (pw, pw), (0, 0)], 'constant')
        self.ph, self.pw = ph, pw
        image_nor = normalization(image_pad, black_level, white_level)
        image_nchw= np.expand_dims(image_nor, axis = 0).transpose(0,3,1,2)
        image_tensor = torch.from_numpy(image_nchw).float()  
        name = self.test_list[index]

        return image_tensor, self.ph, self.pw, height, width, name


    def __len__(self):
        return len(self.test_list)


    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] == 'DS_Store':
                continue     
            if splits[-1] not in ['dng']:
                raise ValueError




def train_dataloader(path, batch_size, num_workers):

    dataloader = DataLoader(
        DngTrainset(os.path.join(path, 'train')),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader



def valid_dataloader(path, batch_size=1, num_workers=0):

    dataloader = DataLoader(
        DngValidset(os.path.join(path, 'valid')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader



def test_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DngTestset(path), 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True
    )
    return dataloader
