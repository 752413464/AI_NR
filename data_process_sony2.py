
import os
import numpy as np
import rawpy
import cv2
import imageio
import h5py
import torch
from bay_aug import bayer_aug, bayer_unify
import torch.distributions as dist


def normalization(input_data, black_level, white_level):
    
    output_data = np.maximum(input_data.astype(float) - black_level, 0) / (white_level - black_level)
   
    return output_data



def inv_normalization(input_data, black_level, white_level):

    output_data = np.clip(input_data, 0., 1.) * 65535
    output_data = output_data.astype(np.uint16)

    return output_data

def pg_noise3(image_tensor):

  log_min_shot_noise = torch.log10(torch.Tensor([0.00068674]))
  log_max_shot_noise = torch.log10(torch.Tensor([0.02194856]))
  distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)
  log_shot_noise = distribution.sample()

  shot_noise = torch.pow(10, log_shot_noise)

  distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.36]))
  read_noise = distribution.sample()

  line = lambda x: 1.85 * x + 0.2 ### Line SIDD test set
  log_read_noise = line(log_shot_noise) + read_noise

  read_noise = torch.pow(10,log_read_noise)

  image_shot = torch.poisson(image_tensor / shot_noise) * shot_noise

  img_noise = image_shot + torch.FloatTensor(image_shot.shape).normal_(0.0, read_noise.item())

  img_noise = np.clip(0, 1, img_noise)

  var = img_noise * shot_noise + read_noise
  print(shot_noise, read_noise)

  return img_noise, var


def pg_noise(image_tensor):

    log_min_shot_noise = torch.log10(torch.Tensor([0.00018674]))
    log_max_shot_noise = torch.log10(torch.Tensor([0.02194856]))
    distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)
    log_shot_noise = distribution.sample()
    shot_noise = torch.pow(10, log_shot_noise)
    # distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.20]))
    # read_noise = distribution.sample()
    a = np.random.uniform(0.6, 2.0)
    b = np.random.uniform(-3.0, -2.2)
    line = lambda x: a * x + b
    log_read_noise = line(log_shot_noise) + np.random.normal(0.0, 0.62)

    read_noise = torch.pow(10, log_read_noise)

    image_shot = torch.poisson(image_tensor / shot_noise) * shot_noise

    img_noise = image_shot + torch.FloatTensor(image_shot.shape).normal_(0.0, np.sqrt(read_noise.item()))

    img_noise = np.clip(0,1, img_noise)

    var = img_noise * shot_noise + read_noise

    print(shot_noise, read_noise )

    return img_noise, var


def pg_noise2(image_tensor, k , read_scale):

    image_shot = torch.poisson(image_tensor / k) * k

    img_noise = image_shot +  torch.FloatTensor(image_tensor.shape).normal_(0.0, read_scale)

    img_noise = np.clip(0,1, img_noise)

    var = img_noise * k + read_scale

    return img_noise, var


def read_image_train(input_path):

    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible.astype(np.float32)
    bayer_pattern_matrix = raw.raw_pattern
    target_bayer = 'RGGB'
    bayer_desc = 'RGBG'
    bayer_pattern = ''

    for i in range(2):
        for k in range(2):
         bayer_pattern += (bayer_desc[bayer_pattern_matrix[i][k]])

    raw_data = bayer_unify(raw_data, bayer_pattern, target_bayer, 'pad')
    raw_data = bayer_aug(raw_data, input_pattern = target_bayer)

    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                            raw_data_expand[0:height:2, 1:width:2, :],
                                            raw_data_expand[1:height:2, 0:width:2, :],
                                            raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    raw.close()

    return raw_data_expand_c, bayer_pattern, target_bayer


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

    bayer_pattern_matrix = raw.raw_pattern
    target_bayer = 'RGGB'
    bayer_desc = 'RGBG'
    bayer_pattern = ''

    for i in range(2):
        for k in range(2):
         bayer_pattern += (bayer_desc[bayer_pattern_matrix[i][k]])
    print('input_bayer:' , bayer_pattern, 'target_bayer:', target_bayer)
    raw_data = bayer_unify(raw_data, bayer_pattern, target_bayer, 'pad')

    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    return raw_data_expand_c, bayer_pattern, target_bayer



def write_back_tiff(dest_path, raw_data, black_level, white_level, b_p, t_p):

    CCM = [[ 1.9435506 , -0.7152609 , -0.2282897  ],
       [-0.22348748,  1.4704359 , -0.24694835   ],
       [ 0.03258422, -0.69400704,  1.6614228   ]]

    wb_gain = [1.89453125 , 1.0, 1.0 , 2.55078125]
    # wb_gain = [2.55078125 , 1.0, 1.0 , 1.89453125]

    gamma = 2.2

    raw_data = raw_data.clip(0., 1.)

    raw_gain = raw_data * wb_gain

    bayer_inv = inv_normalization(raw_gain, black_level, white_level)

    bayer_inv = bayer_inv.reshape(bayer_inv.shape[1], bayer_inv.shape[2], 2, 2).transpose(0, 2, 1, 3).reshape(bayer_inv.shape[1]*2, bayer_inv.shape[2]*2)

    bayer_inv = bayer_unify(bayer_inv, b_p, t_p, 'crop')

    bayer_rgb = cv2.cvtColor(bayer_inv, cv2.COLOR_BAYER_RG2BGR).astype(np.float32) / 65535

    rgb_ccm = bayer_rgb.dot(np.array(CCM).T).clip(0, 1)

    rgb_gamma = rgb_ccm ** (1 / gamma)

    rgb = inv_normalization(rgb_gamma, black_level, white_level)

    tiff_path = dest_path.split('.')[0] + '.tiff'

    imageio.imsave(tiff_path, rgb)


def write_image(input_data, height, width):

    output_data = input_data.reshape(height, width, 2, 2).transpose(0, 2, 1, 3).reshape(height*2, width*2)

    return output_data
