
import os
import numpy as np
import rawpy
import cv2
import imageio


def normalization(input_data, black_level, white_level):
    
    output_data = np.maximum(input_data.astype(float), 0) / 65535
   
    return output_data



def inv_normalization(input_data, black_level, white_level):

    output_data = np.clip(input_data, 0., 1.) * 65535
    output_data = output_data.astype(np.uint16)

    return output_data


def write_back_tiff(dest_path, raw_data, h, w , black_level, white_level ):

    CCM =[[1.639876127243042, -0.3148389160633087, -0.3250371813774109], 
       [-0.16239236295223236, 1.3735886812210083, -0.21119630336761475],
        [-0.022998422384262085, -0.32546359300613403, 1.3484619855880737]]

    wb_gain = [1.432167887687683, 1.0, 1.0, 2.160337448120117]

    gamma = 2.2

    raw_data = raw_data.clip(0., 1.)

    raw_gain = raw_data * wb_gain

    bayer = write_image(raw_gain, h, w)

    bayer_inv = inv_normalization(bayer, black_level, white_level)

    bayer_rgb = cv2.cvtColor(bayer_inv, cv2.COLOR_BAYER_BG2BGR).astype(np.float32) /65535

    rgb_ccm = bayer_rgb.dot(np.array(CCM).T).clip(0, 1)

    rgb_gamma = rgb_ccm ** (1 / gamma)
   
    rgb = inv_normalization(rgb_gamma, black_level, white_level)

    imageio.imsave(dest_path, rgb)



def read_image(input_path):

    height = 3000
    width = 4000
    array= np.fromfile(input_path,dtype = np.uint16)
    image = array.reshape([height,width])
    image_rggb = image[::-1, ::-1]
    raw_data_expand = np.expand_dims(image_rggb, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    
    return raw_data_expand_c

def write_image(input_data, height, width):

    output_data = input_data.reshape(height, width, 2, 2).transpose(0, 2, 1, 3).reshape(height*2, width*2)
    output_bggr = output_data[::-1, ::-1]
    return output_bggr
