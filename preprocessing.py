# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:28:04 2018

@author: Dragon
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
