
import os
from re import L
import numpy as np
import rawpy
import cv2
import imageio
import torch
import torch.distributions as tdist
from typing import Tuple


class KSigma:
    def __init__(self, k_coeff, b_coeff, anchor: float, v: float = 15871.0):
        
        self.K = np.poly1d(k_coeff)
        self.Sigma = np.poly1d(b_coeff)
        self.anchor = anchor
        self.v = v


    def __call__(self, img_01, iso: float, inverse=False):

        img = img_01 * self.v

        k, sigma = self.K(iso), self.Sigma(iso)

        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)

        cvt_k = k_a / k
        
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        if not inverse:

            img = img * cvt_k + cvt_b


        else:
            # print('img_out = ', img.max())

            img = (img - cvt_b) / cvt_k

            # print('img_ink = ', img.max())



        # if not inverse:
        #     img = img / k + (sigma / k ** 2)
        # else:
        #     img = (img + (sigma / k ** 2)) * k
       
        return img / self.v

class KSigma_reno:

    def __init__(self, K_coeff, B_coeff, anchor: float, V: float = 959.0):
        self.K = np.poly1d(K_coeff)
        self.Sigma = np.poly1d(B_coeff)
        self.anchor = anchor
        self.V = V

    def __call__(self, img_01, iso: float, inverse=False):
        k, sigma = self.K(iso), self.Sigma(iso)
        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        img = img_01 * self.V

        if not inverse:

            # print('img_in = ', img.max())

            img = img * cvt_k + cvt_b

            # print('img_k = ', img.max())


        else:
            # print('img_out = ', img.max())

            img = (img - cvt_b) / cvt_k

            # print('img_ink = ', img.max())

        return img / self.V


class KSigma_tensor:
    def __init__(self, device, k_coeff: Tuple[float, float], b_coeff: Tuple[float, float, float], anchor: float, v: float = 15871.0):
        
        self.K = np.poly1d(k_coeff)
        self.Sigma = np.poly1d(b_coeff)
        self.anchor = anchor
        self.v = v
        self.device = device



    def __call__(self, img_01, iso: float, inverse=False):

        img = img_01 * self.v

        k, sigma = self.K(iso), self.Sigma(iso)

        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)

        cvt_k = k_a / k

        
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a


        cvt_k = torch.from_numpy(cvt_k).float().to(self.device)

        cvt_b = torch.from_numpy(cvt_b).float().to(self.device)


        if not inverse:
            img = torch.mul(img, cvt_k)
            img = torch.add(img, cvt_b)

        else:
            img = torch.sub(img, cvt_b)
            img = torch.div(img, cvt_k)
        
        # if not inverse:
        #     img = img / k + (sigma / k ** 2)
        # else:
        #     img = (img + (sigma / k ** 2)) * k
       
        return torch.div(img, self.v)




def normalization(input_data, black_level, white_level):
    
    output_data = np.maximum(input_data.astype(float) - black_level, 0) / (white_level - black_level)
   
    return output_data



def inv_normalization(input_data, black_level, white_level):

    output_data = np.clip(input_data, 0., 1.) * 65535

    output_data = output_data.astype(np.uint16)

    return output_data


def generate_poisson_(y, k):

    y = torch.poisson(y / k) * k
    return y


def generate_read_noise(shape, scale, loc=0):

    read = torch.FloatTensor(shape).normal_(loc, scale)

    return read


def pg_noise(clean_tensor, k, read_scale):

    noisy_shot = generate_poisson_(clean_tensor, k)

    read_noise = generate_read_noise(clean_tensor.shape, scale=read_scale)

    noisy = noisy_shot + read_noise

    return noisy


def inv_k(data, iso, camera):

    if camera == 'gp':

        k_coeff = [2.19586075e-06, 8.43112069e-05]

        b_coeff = [1.11039570e-12, 8.57579026e-10, 2.15851893e-06]

    elif camera == 's6':

        k_coeff = [4.48916546e-06, 2.02585627e-03]

        b_coeff = [2.27373832e-13, 5.53331696e-09, 6.10968591e-07]

    elif camera == 'ip':

        k_coeff = [2.74289431e-07, 9.99170650e-04]
        
        b_coeff = [ 3.97667623e-21, -1.98173628e-17, 3.46822012e-14, -2.55887922e-11,
                7.74059183e-09, -8.11401512e-08]

    elif camera == 'sy':
        
        k_coeff =  [5.86116626e-08,1.77486337e-05]

        b_coeff = [-1.15787697e-21, -2.53544295e-18, -1.28930515e-15, 2.97640275e-11, 1.76379864e-07, 9.72987856e-05]
    
    elif camera == 're':
            
        k_coeff  = [2.35108510e-06, 3.40729804e-05]
 
        b_coeff = [1.08609445e-11, 9.94115206e-09, 1.75367559e-06]


    k_sigma = KSigma(
           
            k_coeff = k_coeff,
            b_coeff = b_coeff,
            anchor = 1600
        )

    data_inv_k  = k_sigma(data, iso,  inverse = True)

    return data_inv_k


def inv_k_reno(data, iso, camera):

    if camera == 'gp':

        k_coeff = [2.19586075e-06, 8.43112069e-05]

        b_coeff = [1.11039570e-12, 8.57579026e-10, 2.15851893e-06]

    elif camera == 's6':

        k_coeff = [4.48916546e-06, 2.02585627e-03]

        b_coeff = [2.27373832e-13, 5.53331696e-09, 6.10968591e-07]

    elif camera == 'ip':

        k_coeff = [2.74289431e-07, 9.99170650e-04]
        
        b_coeff = [ 3.97667623e-21, -1.98173628e-17, 3.46822012e-14, -2.55887922e-11,
                7.74059183e-09, -8.11401512e-08]

    elif camera == 'sy':
        
        k_coeff =  [5.86116626e-08, 1.77486337e-05]

        b_coeff = [-1.15787697e-21, -2.53544295e-18, -1.28930515e-15, 2.97640275e-11, 1.76379864e-07, 9.72987856e-05]
    
    elif camera == 're':
            
        k_coeff  = [2.35108510e-06, 3.40729804e-05]
 
        b_coeff = [1.08609445e-11, 9.94115206e-09, 1.75367559e-06]


    k_sigma = KSigma_reno(
            k_coeff = k_coeff,
            b_coeff = b_coeff,
            anchor = 1600
        )

    data_inv_k  = k_sigma(data, iso,  inverse = True)

    return data_inv_k


def inv_k_tensor(data, iso, device, camera):

    if camera == 'gp':

        k_coeff = [2.19586075e-06, 8.43112069e-05]

        b_coeff = [1.11039570e-12, 8.57579026e-10, 2.15851893e-06]

    elif camera == 's6':

        k_coeff = [4.48916546e-06, 2.02585627e-03]

        b_coeff = [2.27373832e-13, 5.53331696e-09, 6.10968591e-07]

    elif camera == 'ip':

        k_coeff = [2.74289431e-07, 9.99170650e-04]
        
        b_coeff = [ 3.97667623e-21, -1.98173628e-17, 3.46822012e-14, -2.55887922e-11,
                7.74059183e-09, -8.11401512e-08]

    elif camera == 'sy':
        
        k_coeff =  [5.86116626e-08,1.77486337e-05]

        b_coeff = [-1.15787697e-21, -2.53544295e-18, -1.28930515e-15, 2.97640275e-11, 1.76379864e-07, 9.72987856e-05]
    
    
    elif camera == 're':
            
        k_coeff  = [2.35108510e-06, 3.40729804e-05]
 
        b_coeff = [1.08609445e-11, 9.94115206e-09, 1.75367559e-06]

    k_sigma = KSigma_tensor(
            device,
            k_coeff = k_coeff,
            b_coeff = b_coeff,
            anchor = 1600,
        )



    data_inv_k  = k_sigma(data, iso,  inverse = True)

    return data_inv_k



def add_noise(gt_image, k , read_scale ):

    variance = gt_image * k + read_scale

    n        = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance)) 
   
    noise    = n.sample()
    
    out      = gt_image + noise
    
    return out


def write_back_tiff(dest_path, raw_data, h, w , black_level, white_level ):

    CCM = [[ 1.9435506 , -0.7152609 , -0.2282897  ],
       [-0.22348748,  1.4704359 , -0.24694835   ],
       [ 0.03258422, -0.69400704,  1.6614228   ]]

    wb_gain = [1.89453125 , 1.0, 1.0 , 2.55078125]

    gamma = 2.2

    raw_data = raw_data.clip(0., 1.)

    # raw_data =  normalization(raw_data , black_level, white_level)

    raw_gain = raw_data * wb_gain

    bayer = write_image(raw_gain, h, w)

    bayer_inv = inv_normalization(bayer, black_level, white_level)

    bayer_rgb = cv2.cvtColor(bayer_inv, cv2.COLOR_BAYER_RG2BGR).astype(np.float32) / 65535

    rgb_ccm = bayer_rgb.dot(np.array(CCM).T).clip(0, 1)

    rgb_gamma = rgb_ccm ** (1 / gamma)
   
    rgb = inv_normalization(rgb_gamma, black_level, white_level)

    tiff_path = dest_path.split('.')[0] + '.tiff'

    imageio.imsave(tiff_path, rgb)



def unpack_image(raw_data):

    height = raw_data.shape[1]

    width = raw_data.shape[2]

    raw_data_expand_c = np.concatenate((raw_data[:, 0:height:2, 0:width:2],
                                        raw_data[:, 0:height:2, 1:width:2],
                                        raw_data[:, 1:height:2, 0:width:2],
                                        raw_data[:, 1:height:2, 1:width:2]), axis=0)
    raw_data_expand = np.expand_dims(raw_data_expand_c, axis=0)

    raw_data_expand = torch.from_numpy(raw_data_expand).float()

    return raw_data_expand



def write_image(input_data, height, width):

    output_data = input_data.reshape(height, width, 2, 2).transpose(0, 2, 1, 3).reshape(height*2, width*2)

    return output_data