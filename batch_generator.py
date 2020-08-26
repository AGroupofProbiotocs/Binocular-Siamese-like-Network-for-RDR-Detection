# -*- coding: utf-8 -*-
"""
========================================================================
A siamese-like CNN for diabetic retinopathy detection using binocular 
fudus images as input, Version 1.0
Copyright(c) 2020 Xianglong Zeng, Haiquan Chen, Yuan Luo, Wenbin Ye
All Rights Reserved.
----------------------------------------------------------------------
Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is here
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation. This program
shall not be used, rewritten, or adapted as the basis of a commercial
software or hardware product without first obtaining permission of the
authors. The authors make no representations about the suitability of
this software for any purpose. It is provided "as is" without express
or implied warranty.
----------------------------------------------------------------------
Please cite the following paper when you use it:
X. Zeng, H. Chen, Y. Luo and W. Ye, "Automated Diabetic Retinopathy 
Detection Based on Binocular Siamese-Like Convolutional Neural 
Network," in IEEE Access, vol. 7, pp. 30744-30753, 2019, 
doi: 10.1109/ACCESS.2019.2903171.
----------------------------------------------------------------------
This file defines the generator to generate the data batches.

@author: Xianglong Zeng
========================================================================
"""
from inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from preprocessing import retinal_img_preprocessing
from numpy import random
from PIL import Image
import os
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import cv2


def center_crop_PIL(img, center_crop_size):
    # img is a PIL instange
    # 按中心剪裁
    width, height = img.size
    w, h = center_crop_size
    left = (width - w) // 2
    top = (height - h) // 2
    right = left + w
    bottom = top + h
    return img.crop((left, top, right, bottom))


def scale_byRatio(img_path, ratio=1.0, return_width=299, crop_method=center_crop_PIL):
    # Given an image path, return a scaled array
    # 载入图片
    img = load_img(img_path)

    # 将图片根据ratio按比例放缩
    w, h = img.size
    w = int(ratio * w)
    h = int(ratio * h)

    # 将图片resize，使用抗锯齿
    img = img.resize((w, h), Image.ANTIALIAS)

    shorter = min(w, h)
    longer = max(w, h)

    shorter_side = return_width
    longer_side = int(longer * 1. / shorter * shorter_side)

    img_width = shorter_side if w < h else longer_side
    img_height = longer_side if w < h else shorter_side

    # 将图片根据短边按比例resize
    img = img.resize((img_width, img_height), Image.ANTIALIAS)

    # 将图片按中心位置剪裁至299x299
    img_cropped = crop_method(img, (return_width, return_width))

    return img_cropped


# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
chance = 0.5
seq = iaa.Sequential([
    iaa.Sometimes(chance,[iaa.Fliplr(1),iaa.Flipud(1)]), # horizontally flip 50% of all images
    iaa.Sometimes(chance, iaa.Crop(percent=(0, 0.05))),  # crop images by 0-10% of their height/width
    iaa.Sometimes(chance, iaa.Add((-10, 10), per_channel=0.5)),# change brightness of images (by -10 to 10 of original value)
    iaa.Sometimes(chance, iaa.Multiply((0.85, 1.15), per_channel=0.5)),# change brightness of images (85-115% of original value)
    iaa.Sometimes(chance, iaa.ContrastNormalization((0.85, 1.15), per_channel=0.5)),  # improve or worsen the contrast
    iaa.Sometimes(chance, iaa.Affine(
        scale={"x": (1.0, 1.1), "y": (1.0, 1.1)},  # scale images to 90-110% of their size, individually per axis
        translate_px={"x": (-5, 5), "y": (-5, 5)}, # translate by -16 to +16 pixels (per axis)
        rotate=(-30, 15),  # rotate by -180 to +180 degrees
        shear=(-10, 10),  # shear by -10 to +10 degrees
        order=ia.ALL,  # use any of scikit-image's interpolation methods
        # mode='edge' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    ))
],
    random_order=True  # do all of the above in random order
)


def generator_img_batch(data_list, nbr_classes=1, batch_size=32, return_label=True, hsv=False,
                        crop_method=center_crop_PIL, preprocess=False, img_width=299, img_height=299,
                        random_shuffle=True, save_to_dir=None, augment=False, call_counts='',
                        normalize=True, crop=True, mirror=False):
    '''
    A generator that yields a batch of (data, label).

    Input:
        data_list  : a tuple contains of two lists of binoculus data, e.g.
                     ("/data/workspace/dataset/Cervical_Cancer/train/10_left.jpg 0",
                      "/data/workspace/dataset/Cervical_Cancer/train/10_rightt.jpg 0")
        sf   : whether shuffle rows in the data_llist
        batch_size : batch size

    Output:
        (X_batch, Y_batch)
    '''

    # 固定当前随机状态
    seq_fixed = seq.to_deterministic()
    randnum = np.int(100*random.random())
    # print('HI~')

    left_data_list, right_data_list = data_list

    N = len(left_data_list)

    if random_shuffle:
        random.seed(randnum)
        random.shuffle(left_data_list)
        random.seed(randnum)
        random.shuffle(right_data_list)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        # 判断是否到达最后一个batch，若是则修改该batch的size
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        X_left_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        X_right_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_left_batch = np.zeros((current_batch_size, nbr_classes))
        Y_right_batch = np.zeros((current_batch_size, nbr_classes))

        for i in range(current_index, current_index + current_batch_size):
            line_left = left_data_list[i].strip().split(' ')
            label_left = int(line_left[-1])
            img_path_left = line_left[0]
            line_right = right_data_list[i].strip().split(' ')
            label_right = int(line_right[-1])
            img_path_right = line_right[0]

            # 将图片按比例缩放并剪裁，方便后面拼接为batch
            if crop:
                left_img = scale_byRatio(img_path_left, return_width=img_width, crop_method=crop_method)
                right_img = scale_byRatio(img_path_right, return_width=img_width, crop_method=crop_method)
            else:
                left_img = load_img(img_path_left)
                right_img = load_img(img_path_right)
            left_img = img_to_array(left_img)
            right_img = img_to_array(right_img)

            if hsv:
                # Change from RGB space to HSV space
                left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2HSV)
                right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2HSV)
                # Mapping to [0, 255]
                left_img = np.interp(left_img, [left_img.min(), left_img.max()], [0, 255])
                right_img = np.interp(right_img, [right_img.min(), right_img.max()], [0, 255])

            # 将当前图像填装如batch
            X_left_batch[i - current_index] = left_img
            X_right_batch[i - current_index] = right_img
            ##将标签转换为one-hot形式,仅在多分类时使用
            #Y_left_batch[i - current_index, label_left] = 1
            #Y_right_batch[i - current_index, label_right] = 1
            Y_left_batch[i - current_index] = label_left
            Y_right_batch[i - current_index] = label_right

        if mirror:
            if random.random() < 0.5:
                X_left_batch, X_right_batch = X_right_batch, X_left_batch
                Y_left_batch, Y_right_batch = Y_right_batch, Y_left_batch
                X_left_batch = iaa.Fliplr(1).augment_images(X_left_batch)
                X_right_batch = iaa.Fliplr(1).augment_images(X_right_batch)

        if augment:
            X_left_batch = seq_fixed.augment_images(X_left_batch)
            X_right_batch = iaa.Fliplr(1).augment_images(X_right_batch)
            X_right_batch = seq_fixed.augment_images(X_right_batch)
            X_right_batch = iaa.Fliplr(1).augment_images(X_right_batch)

        X_left_batch = X_left_batch.astype(np.uint8)
        X_right_batch = X_right_batch.astype(np.uint8)

        # 眼底图片预处理
        if preprocess:
            for i in range(current_batch_size):
                X_left_batch[i] = retinal_img_preprocessing(X_left_batch[i], return_image=True,
                                                            result_size=(img_width, img_height))
                X_right_batch[i] = retinal_img_preprocessing(X_right_batch[i], return_image=True,
                                                             result_size=(img_width, img_height))

        # 导出扩充后的图片数据集
        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_path_left = left_data_list[i].strip().split(' ')[0]
                tmp_path_right = right_data_list[i].strip().split(' ')[0]
                image_name_left = call_counts + tmp_path_left.split(os.sep)[-1]
                image_name_right = call_counts + tmp_path_right.split(os.sep)[-1]
                #                image_name = '_'.join(basedir)
                img_to_save_path_left = os.path.join(save_to_dir, image_name_left)
                img_to_save_path_right = os.path.join(save_to_dir, image_name_right)

                img_left = array_to_img(X_left_batch[i - current_index])
                img_right = array_to_img(X_right_batch[i - current_index])

                img_left.save(img_to_save_path_left)
                img_right.save(img_to_save_path_right)

        if normalize:
            X_left_batch = X_left_batch.astype(np.float64)
            X_right_batch = X_right_batch.astype(np.float64)
            X_left_batch = preprocess_input(X_left_batch)
            X_right_batch = preprocess_input(X_right_batch)

        X_batch = {'left_input': X_left_batch, 'right_input': X_right_batch}
        Y_batch = {'left_output': Y_left_batch, 'right_output': Y_right_batch}

        if return_label:
            yield (X_batch, Y_batch)
        else:
            yield (X_left_batch, X_right_batch)