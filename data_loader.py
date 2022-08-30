import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from dataloader.data_process_reno import normalization, read_image
import glob



class DngTrainset(Dataset):

    def __init__(self):

        train_fns = glob.glob('PMRID_data/*/dark/*')
        
        train_id = len(train_fns) * 0.8

        self.train_fns = train_fns[:int(train_id)]

   
    def __len__(self):
        return len(self.train_fns)

    def __getitem__(self, idx):
    
        black_level = 512
        white_level = 16383

        id = self.train_fns[idx]

        gt_files = glob.glob('%s'%id + '/' + 'gt*')[-1]

        in_files = glob.glob('%s'%id + '/' + 'input*')[-1]

        image= read_image(in_files)

        image = normalization(image, black_level, white_level)
            
        image_nchw= np.expand_dims(image, axis = 0).transpose(0,3,1,2)

        image_tensor = torch.from_numpy(image_nchw).float()

        H , W = image.shape[0:2]

        label= read_image(gt_files)

        label = normalization(label, black_level, white_level)

        label_nchw = np.expand_dims(label , axis = 0).transpose(0,3,1,2)

        label_tensor = torch.from_numpy(label_nchw).float()
 

        ##add noise
        # noise = random.random() < 0.5
        # if noise:        
        #     # shot_noise, read_noise = random_noise_levels()
        #     # image  = add_noise(image, shot_noise, read_noise)
        #     image = add_noise(image)

        return image_tensor, label_tensor, H, W


class DngValidset(Dataset):

    def __init__(self):

        valid_fns = glob.glob('PMRID_data/' + '/*/dark/*')

        train_id = len(valid_fns) * 0.8
        
        valid_id = len(valid_fns) * 0.9
        
        self.valid_fns = valid_fns[int(train_id):int(valid_id)]


    def __len__(self):

        return len(self.valid_fns)

    def __getitem__(self, idx):

        black_level = 512
        white_level = 16383

        id = self.valid_fns[idx]

        gt_files = glob.glob('%s'%id + '/' + 'gt*')[-1]

        in_files = glob.glob('%s'%id + '/' + 'input*')[-1]

        image= read_image(in_files)

        image = normalization(image, black_level, white_level)
            
        image_nchw= np.expand_dims(image, axis = 0).transpose(0,3,1,2)

        image_tensor = torch.from_numpy(image_nchw).float()

        H , W = image.shape[0:2]

        label= read_image(gt_files)

        label = normalization(label, black_level, white_level)

        label_nchw = np.expand_dims(label , axis = 0).transpose(0,3,1,2)

        label_tensor = torch.from_numpy(label_nchw).float()

      
        return image_tensor, label_tensor, H, W


class DngTestset(Dataset):

    def __init__(self):

        self.gt_fns = glob.glob('Reno/' + 'gt*')
        self.input_fns= glob.glob('Reno/' + 'input*')
        print('Test examples = ',  len(self.gt_fns))


    def __len__(self):
        return len(self.gt_fns)
    
    def __getitem__(self, idx):

        black_level = 512
        white_level = 16383

        gt_path = self.gt_fns[idx]

        input_path = self.input_fns[idx]
       

        gt_image = read_image(gt_path)
        label = normalization(gt_image, black_level, white_level)

        input_image = read_image(input_path)
        image = normalization(input_image, black_level, white_level)

        H , W = image.shape[0:2]
  
        image_nchw= np.expand_dims(image, axis = 0).transpose(0,3,1,2)
        image_tensor = torch.from_numpy(image_nchw).float()

        label_nchw = np.expand_dims(label, axis = 0).transpose(0,3,1,2)
        label_tensor = torch.from_numpy(label_nchw).float()
      
        return image_tensor, label_tensor, H, W, input_path


def train_dataloader( batch_size, num_workers):

    dataloader = DataLoader(
        DngTrainset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True
    )
    return dataloader



def valid_dataloader(batch_size=1, num_workers=1):

    dataloader = DataLoader(
        DngValidset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader


def test_dataloader(batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DngTestset(), 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True
    )
    return dataloader
