# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.08.22'

import numpy as np
import model
import cv2

dir = '/home/smher/Documents/Hyperspectrals/20180822/hs_img_datas_0001.raw'


def read_row_data(file_name=dir):
    pass


if __name__ == '__main__':
    fd = open(dir, 'rb')
    rows = 587
    cols = 696
    bin_img = np.fromfile(fd, dtype=np.uint8)
    #im = bin_img.reshape((rows, cols))
    im = bin_img
    fd.close()
    print('type of im: ', type(im))    # numpy.ndarray
    print('shape of im: ', im.shape)   # (587, 696)

    '''
    img = read_row_data()
    rgb_img = model.show_hyper_img_top(img, True)
    print('Test passed.')
    '''
