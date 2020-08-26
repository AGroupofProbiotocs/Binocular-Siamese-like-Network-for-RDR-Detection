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
This file defines the preprocessing function.

@author: Xianglong Zeng
========================================================================
"""

import cv2, numpy

def scaleRadius(img, scale):
    x = img[int((img.shape[0])/2), :, : ].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    if r == 0:
        s = 2
    else:
        s = scale * 1.0 / r
        if s>=10 or s<=1:
            s = 2
    return cv2.resize(img, None, fx = s, fy = s)


def retinal_img_preprocessing(img, result_path=None, return_image=True, save_result=False,
                          show_result=False, result_size=(299, 299), scale=300):
    '''
    :param result_path:  the path where the result is saved, default is None
    :param return_image: Flag, when it is true, direcly return the processed image
    :param save_result: Flag, when it is true, the result will be saved. default is True
    :param show_result: Flag, when it is true, the result will be showed in screen. default is False
    :param result_size: the size of result image, e.g. (299,299)
    :return:
    '''
    # parameters
    a = img
    a = scaleRadius(a, scale)
    a = cv2.addWeighted (a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128 )
    #remove outer 10%
    b = numpy.zeros(a.shape)
    cv2.circle(b, (int(a.shape[1] / 2), int(a.shape[0] / 2)), int(scale * 0.95), (1, 1, 1), -1, 8, 0)
    a = a * b + 128 * (1 - b)
#    #???不知道是什么操作
#    width = a.shape[0]
#    height = a.shape[1]
#    if width > height:
#        a = a[(width - height) / 2 : width - (width - height) / 2, :]
#        width = a.shape[0]
#        a = a[int(width * 0.03):int(width * 0.97)] 
#
#    elif width < height:
#        a = a[:, int((height - width) / 2) : int(height - (height - width) / 2)]
#        height = a.shape[1]
#        a = a[:, int(height * 0.03):int(height * 0.97)]
#
    if result_size is not None:
        a = cv2.resize(a, result_size)

    if return_image:
        return a

    if show_result:
        cv2.imshow('processed image', a)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_result:
        cv2.imwrite(result_path,a)
