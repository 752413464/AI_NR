
import os
import numpy as np
import rawpy
import cv2
import imageio
import h5py
from dataloader.bay_aug import bayer_aug



def normalization(input_data, black_level, white_level):
    
    output_data = np.maximum(input_data.astype(float) - black_level, 0) / (white_level - black_level)
   
    return output_data



def inv_normalization(input_data, black_level, white_level):

    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)

    return output_data

def read_image_train(input_path, h_pro, w_pro):

    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible.astype(np.float32) 
    raw_data = bayer_aug(raw_data, h_pro, w_pro,False, input_pattern='RGGB')
   
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                            raw_data_expand[0:height:2, 1:width:2, :],
                                            raw_data_expand[1:height:2, 0:width:2, :],
                                            raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    raw.close()
    return raw_data_expand_c


def write_back_dng(src_path, dest_path, raw_data):
    # (3472, 4624)
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)  # 获取文件大小
    data_len = width * height * 2
    header_len = 8

    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()

    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")


def read_image(input_path):
    raw = rawpy.imread(input_path)

    raw_data = raw.raw_image_visible.astype(np.float32) 

    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    return raw_data_expand_c




def write_image(input_data, height, width):

    output_data = input_data.reshape(height, width, 2, 2).transpose(0, 2, 1, 3).reshape(height*2, width*2)

    return output_data
