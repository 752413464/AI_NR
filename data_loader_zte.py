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
from dataloader.data_process_zte import normalization, read_image,read_image_train
import random
import glob



class DngTrainset(Dataset):

    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.input_list = os.listdir(os.path.join(image_dir, 'noisy'))
        self.ground_list = os.listdir(os.path.join(image_dir, 'ground_truth'))
        
    def __len__(self):
        # self.input_list = self.input_list[:14]
        return len(self.input_list)

    def __getitem__(self, idx):
    
        black_level = 1024
        white_level = 16383

        h_pro = np.random.random()
        w_pro = np.random.random()

        image = read_image_train(os.path.join(self.image_dir, 'noisy', self.input_list[idx]), h_pro, w_pro)
        image = normalization(image, black_level, white_level)
        label= read_image_train(os.path.join(self.image_dir, 'ground_truth', self.ground_list[idx]),  h_pro, w_pro)
        label = normalization(label, black_level, white_level)

        H , W = image.shape[0:2]

        # patch_x = 512
        # patch_y = 512


        # xx = np.random.randint(0,(width/2) - patch_x)
        # yy = np.random.randint(0,(height/2) -patch_y)

        # image_patch = image[yy:yy+patch_y,xx:xx+patch_x,:]
        # label_patch = label[yy:yy+patch_y,xx:xx+patch_x,:]

  
        image_nchw= np.expand_dims(image, axis = 0).transpose(0,3,1,2)
        image_tensor = torch.from_numpy(image_nchw).float()

        # image_tensor = torch.from_numpy(np.transpose(image.reshape(-1, height//2, width//2, 4), (0, 3, 1, 2))).float()

        # label_tensor = torch.from_numpy(np.transpose(label.reshape(-1, height//2, width//2, 4), (0, 3, 1, 2))).float()


        label_nchw = np.expand_dims(label, axis = 0).transpose(0,3,1,2)
        label_tensor = torch.from_numpy(label_nchw).float()
 
        ##add noise
        # noise = random.random() < 0.5
        # if noise:        
        #     # shot_noise, read_noise = random_noise_levels()
        #     # image  = add_noise(image, shot_noise, read_noise)
        #     image = add_noise(image)
    
        return image_tensor, label_tensor, H, W

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
        
        # self._check_image(self.input_list)
        # self.input_list.sort()
        # self._check_image(self.ground_list)
        # self.ground_list.sort()
       
       
    def __len__(self):
     
        self.input_list = self.input_list[:]
        return len(self.input_list)

    def __getitem__(self, idx):


        black_level = 1024
        white_level = 16383

        image = read_image(os.path.join(self.image_dir, 'noisy', self.input_list[idx]))
        height, width = image.shape[0:2]
        image = normalization(image, black_level, white_level)


        label = read_image(os.path.join(self.image_dir, 'ground_truth', self.ground_list[idx]))
        label = normalization(label, black_level, white_level)


        image_nchw= np.expand_dims(image, axis = 0).transpose(0,3,1,2)
        image_tensor = torch.from_numpy(image_nchw).float()

        label_nchw = np.expand_dims(label, axis = 0).transpose(0,3,1,2)
        label_tensor = torch.from_numpy(label_nchw).float()

      
        return image_tensor, label_tensor, height, width

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] == 'DS_Store':
                continue
            if splits[-1] not in ['dng']:
                raise ValueError

class DngTestset(Dataset):

    def __init__(self, image_dir):


        self.image_dir = image_dir
        self.input_list = os.listdir(os.path.join(image_dir, 'noisy'))
        self.ground_list = os.listdir(os.path.join(image_dir, 'ground_truth'))
      


    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, idx):

        black_level = 1024
        white_level = 16383
        image_path = os.path.join(self.image_dir, 'noisy', self.input_list[idx])
        label_path = os.path.join(self.image_dir, 'ground_truth', self.ground_list[idx])
        image = read_image(image_path)
        image = normalization(image, black_level, white_level)

        label = read_image(label_path)
        label = normalization(label, black_level, white_level)

        image_nchw= np.expand_dims(image, axis = 0).transpose(0,3,1,2)
        image_tensor = torch.from_numpy(image_nchw).float()

        label_nchw = np.expand_dims(label, axis = 0).transpose(0,3,1,2)
        label_tensor = torch.from_numpy(label_nchw).float()

        name = image_path
        
        H , W = image.shape[0:2]

        return image_tensor, label_tensor, H, W, name

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
        # pin_memory=True
    )
    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=1):

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
