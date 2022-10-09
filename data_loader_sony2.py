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
from data_process_zte import normalization, read_image,read_image_train,pg_noise, pg_noise3, pg_noise2
import random
import glob



class DngTrainset(Dataset):

    def __init__(self, camera):

        # ground_list1 = glob.glob('S:\Data_zte\long//' + '0*.ARW')
        # ground_list2 = glob.glob('S:\Data_zte\long//' + '*.dng')
        # ground_list3 = glob.glob('S:\Data_zte\long//' + '1*.ARW')
        # ground_list = ground_list1 + ground_list2 + ground_list3
        ground_list =  glob.glob('S:\Data_zte\long//' + '*.ARW')
        self.ground_list = ground_list[:215]
        self.camera = camera
        camera_list = ['gp', 's6', 'ip']
        self.camera_list = camera_list

    def __len__(self):

        return len(self.ground_list)

    def __getitem__(self, idx):

        self.camera = self.camera_list[np.random.choice(len(self.camera_list))]
        if self.camera == 'gp':
            self.k_coeff = [2.19586075e-06, 8.43112069e-05]

            self.b_coeff = [1.11039570e-12, 8.57579026e-10, 2.15851893e-06]

        elif self.camera == 's6':

            self.k_coeff = [4.48916546e-06, 2.02585627e-03]

            self.b_coeff = [2.27373832e-13, 5.53331696e-09, 6.10968591e-07]

        elif self.camera == 'ip':

            self.k_coeff = [2.74289431e-07, 9.99170650e-04]

            self.b_coeff = [3.97667623e-21, -1.98173628e-17, 3.46822012e-14, -2.55887922e-11,
                            7.74059183e-09, -8.11401512e-08]

        elif self.camera == 're':

            self.k_coeff = [2.35108510e-06, 3.40729804e-05]

            self.b_coeff = [1.08609445e-11, 9.94115206e-09, 1.75367559e-06]


        black_level = 512
        white_level = 16383

        iso_set = [100, 200, 400, 800, 1600, 3200, 6400]
        # iso = iso_set[np.random.choice(len(iso_set))]
        # iso = int((np.random.random() + 0.01) * 6400)
        iso = int(np.random.uniform(1, 66)) * 100

        k = np.poly1d(self.k_coeff)(iso)

        read_scale = np.poly1d(self.b_coeff)(iso)

        print(iso, self.camera, k, read_scale)

        label , _, _ = read_image_train(self.ground_list[idx])
        label = normalization(label, black_level, white_level)
        H , W = label.shape[0:2]

        label_nchw = np.expand_dims(label, axis = 0).transpose(0,3,1,2)
        label_tensor = torch.from_numpy(label_nchw).float()
        image_tensor, var = pg_noise2(label_tensor, k, read_scale)

        return image_tensor, label_tensor, var, H, W


class DngValidset(Dataset):

    def __init__(self, camera):

        ground_list = glob.glob('S:\Data_zte\long//' + '2*.ARW')
        self.ground_list = ground_list[5:12:2]
        self.camera = camera

        if self.camera == 'gp':

            self.k_coeff = [2.19586075e-06, 8.43112069e-05]

            self.b_coeff = [1.11039570e-12, 8.57579026e-10, 2.15851893e-06]

        elif self.camera == 's6':

            self.k_coeff = [4.48916546e-06, 2.02585627e-03]

            self.b_coeff = [2.27373832e-13, 5.53331696e-09, 6.10968591e-07]


        elif self.camera == 'ip':

            self.k_coeff = [2.74289431e-07, 9.99170650e-04]

            self.b_coeff = [3.97667623e-21, -1.98173628e-17, 3.46822012e-14, -2.55887922e-11,
                            7.74059183e-09, -8.11401512e-08]

        elif self.camera == 'sy':

            self.k_coeff = [5.86116626e-08, 1.77486337e-05]

            self.b_coeff = [-1.15787697e-21, -2.53544295e-18, -1.28930515e-15, 2.97640275e-11, 1.76379864e-07,
                            9.72987856e-05]

        elif self.camera == 're':

            self.k_coeff = [2.35108510e-06, 3.40729804e-05]

            self.b_coeff = [1.08609445e-11, 9.94115206e-09, 1.75367559e-06]
       
    def __len__(self):
        return len(self.ground_list)

    def __getitem__(self, idx):

        black_level = 512
        white_level = 16383
        iso = 3200

        k = np.poly1d(self.k_coeff)(iso)

        read_scale = np.poly1d(self.b_coeff)(iso)

        label, b_p, t_p = read_image_train(self.ground_list[idx])
        height, width = label.shape[0:2]
        label = normalization(label, black_level, white_level)

        label_nchw = np.expand_dims(label, axis = 0).transpose(0,3,1,2)

        label_tensor = torch.from_numpy(label_nchw).float()

        image_tensor, var = pg_noise2(label_tensor, k, read_scale)

        return image_tensor, label_tensor,  var, height, width, b_p, t_p

    # @staticmethod
    # def _check_image(lst):
    #     for x in lst:
    #         splits = x.split('.')
    #         if splits[-1] == 'DS_Store':
    #             continue
    #         if splits[-1] not in ['dng']:
    #             raise ValueError





class DngTestset(Dataset):

    def __init__(self, camera, iso):

        ground_list = glob.glob('S:\Data_zte\long//' + '2*.ARW')
        self.ground_list = ground_list[5:12:2]
        self.camera = camera
        self.iso = iso

        if self.camera == 'gp':

            self.k_coeff = [2.19586075e-06, 8.43112069e-05]

            self.b_coeff = [1.11039570e-12, 8.57579026e-10, 2.15851893e-06]

        elif self.camera == 's6':

            self.k_coeff = [4.48916546e-06, 2.02585627e-03]

            self.b_coeff = [2.27373832e-13, 5.53331696e-09, 6.10968591e-07]


        elif self.camera == 'ip':

            self.k_coeff = [2.74289431e-07, 9.99170650e-04]

            self.b_coeff = [3.97667623e-21, -1.98173628e-17, 3.46822012e-14, -2.55887922e-11,
                            7.74059183e-09, -8.11401512e-08]

        elif self.camera == 'sy':

            self.k_coeff = [5.86116626e-08, 1.77486337e-05]

            self.b_coeff = [-1.15787697e-21, -2.53544295e-18, -1.28930515e-15, 2.97640275e-11, 1.76379864e-07,
                            9.72987856e-05]

        elif self.camera == 're':

            self.k_coeff = [2.35108510e-06, 3.40729804e-05]

            self.b_coeff = [1.08609445e-11, 9.94115206e-09, 1.75367559e-06]


    def __len__(self):
        return len(self.ground_list)
    
    def __getitem__(self, idx):

        black_level = 512
        white_level = 16383

        k = np.poly1d(self.k_coeff)(self.iso)
        read_scale = np.poly1d(self.b_coeff)(self.iso)

        label, b_p, t_p = read_image(self.ground_list[idx])
        label_name = os.path.basename(self.ground_list[idx])
        height, width = label.shape[0:2]
        label = normalization(label, black_level, white_level)
        label_nchw = np.expand_dims(label, axis = 0).transpose(0,3,1,2)
        label_tensor = torch.from_numpy(label_nchw).float()
        # image_tensor = pg_noise(label_tensor)
        image_tensor, var = pg_noise2(label_tensor, k, read_scale)

        return image_tensor, label_tensor,var,  height, width, label_name, b_p, t_p

    # @staticmethod
    # def _check_image(lst):
    #     for x in lst:
    #         splits = x.split('.')
    #         if splits[-1] == 'DS_Store':
    #             continue
    #         if splits[-1] not in ['dng']:
    #             raise ValueError




def train_dataloader(batch_size, num_workers, camera):

    dataloader = DataLoader(
        DngTrainset(camera),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def valid_dataloader( batch_size, num_workers, camera):

    dataloader = DataLoader(
        DngValidset(camera),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader



def test_dataloader( batch_size=1, num_workers=0, camera = 's6', iso = '1600'):
    dataloader = DataLoader(
        DngTestset(camera, iso),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True
    )
    return dataloader
