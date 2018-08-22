# -*- coding: utf-8 -*-

'''
@file: one bit transform for pseudo-color image of hyperspectral input

@brief Reference
  A Low-Complexity Approach for the Color Display of Hyperspectral Remote-Sensing Images
                 Using One-Bit-Transform-Based Band Selection
'''

__author__ = 'smh'
__date__ = '2018.08.22'

import numpy as np
import itertools
import cv2


eps = 1e-6


def _generate_multibandpass_filter(shape=(17, 17)):
    if len(shape) != 2:
        raise ValueError('Shape mismatch. Filter kernel must have a 2d shape.')
    filter_kernel = np.zeros(shape=shape)
    pos = np.array([0, 4, 8, 12, 16])
    for idx in itertools.product(pos, pos):
        filter_kernel[idx[0]][idx[1]] = 1/25
    return filter_kernel


# img is a single band image
def _onebit_transform_band(img, filter_kernel):
    filtered = cv2.filter2D(img, ddepth=3,  kernel=filter_kernel)
    idx = np.where(img > filtered)
    onebit_piexl = np.zeros(img.shape)
    onebit_piexl[idx] = 1

    return onebit_piexl


def _calculate_band_transition(band):
    count_h = 0
    count_w = 0
    H, W = band.shape
    # calculate the transition number along the row
    for i in range(H-1):
        for j in range(W-1):
            count_h += np.bitwise_xor(band[i][j], band[i][j+1])
    # calculate the transition number along the column
    for i in range(W):
        for j in range(H-1):
            count_w += np.bitwise_xor(band[j][i], band[j+1][i])
    return count_h + count_w


# 等价于：计算band1与band2的对应位置的数值进行xor然后加和,因为band1和band2的元素要么为0要么为1
def _calculate_band_correlations(band1, band2):
    return np.sum(np.abs(band1, band2))


# Step 1.1: get the One-Bit representation of image frames.
# calculate 1-bit transform for each band
def _onebittransform(img):
    if len(img.shape) != 3:
        raise ValueError('Shape mismatch(1bt). Input img must have more than one band. Data layout: CHW')
    filter_kernel = _generate_multibandpass_filter()
    obt_lst = []
    for band in img:
        band = _onebit_transform_band(band, filter_kernel)
        obt_lst.append(band)

    return np.array(obt_lst)


# Step 1.2: count the total number of transitions in the horizontal and vertical directions of each band
def _numberoftransitions(img):
    if len(img.shape) != 3:
        raise ValueError('Shape mismatch(transition count). Input img must have more than one band. Data layout: CHW')
    transition_count_lst = []
    for band in img:
        transition_count_lst.append(_calculate_band_transition(band))

    return np.array(transition_count_lst)


# Step 1.3: calculate the local threshold, i.e. the meaning value within wid_size
def _calculate_local_threshold(trans, wid_size):
    if len(trans.shape) != 1:
        raise ValueError('Shape mismatch(local threshold). Input trans must have only one dimension.')
    trans = trans[:, np.newaxis]
    mean_trans = cv2.blur(trans, ksize=(wid_size, 1), borderType=cv2.BORDER_REFLECT)
    mean_trans = np.reshape(mean_trans, newshape=(100))

    return mean_trans


# Step 1: Obtaining Well-Structured Image Bands Using 1BT
def calculate_well_structured(img, filter_shape=(17, 17), wid_size=7, a=0.95):
    obt_img = _onebittransform(img)
    trans_lst = _numberoftransitions(obt_img)
    local_thsh = _calculate_local_threshold(trans_lst, wid_size)
    idx = np.where(local_thsh * a > trans_lst)

    return idx, obt_img


# Step 2.1 calculate the correlation for each well-structured band
# calculate the first two bands
def _calculate_correlation_bands(idx, bands):
    n, h, w = bands.shape
    if n < 3:
        raise ValueError('Shape mismatch(select bands). Input bands should be more than three.')
    least_similar_bands = (0, 0)
    currMaxCorr = 0
    for index in itertools.product(idx, idx):
        id_1 = index[0]
        id_2 = index[1]
        if id_1 == id_2:
            continue
        correlations = _calculate_band_correlations(bands[id_1], bands[id_2])
        if correlations > currMaxCorr:
            currMaxCorr = correlations
            least_similar_bands = index

    return least_similar_bands


def _calculate_third_band(lsb, idx, bands):
    currMaxCorr = 0
    currMaxRatio = 0
    currIdx = idx[0]
    id_1, id_2 = lsb
    for index in idx:
        if index == id_1 or index == id_2:
            continue
        corr1 = _calculate_band_correlations(bands[index], bands[id_1])
        corr2 = _calculate_band_correlations(bands[index], bands[id_2])
        corr = corr1 + corr2
        if corr1 > corr2:
            corrRatio = corr2 / (corr1 + eps)
        else:
            corrRatio = corr1 / (corr2 + eps)
        if currMaxCorr < corr:
            currIdx = index
            currMaxCorr = corr
            currMaxRatio = corrRatio
        elif currMaxCorr == corr:
            if corrRatio > currMaxRatio:
                currIdx = index
                currMaxRatio = corrRatio
            else:
                continue
        else:
            continue

    return currIdx


# Step 2: selecting three suitable bands for the color display
def calculate_rgb(idx, bands):
    id0, id1 = _calculate_correlation_bands(idx, bands)
    id2 = _calculate_third_band((id0, id1), idx, bands)
    var0 = np.std(bands[id0])
    var1 = np.std(bands[id1])
    var2 = np.std(bands[id2])
    var_dict = dict()
    var_dict[var0] = id0
    var_dict[var1] = id1
    var_dict[var2] = id2
    sorted_keys = sorted(var_dict.keys())
    r = bands[var_dict[sorted_keys[0]]]
    g = bands[var_dict[sorted_keys[1]]]
    b = bands[var_dict[sorted_keys[2]]]

    return r, g, b


## @func Top function
## @param img: the input hyperspectral matrix
## @return pseudo_img: the return pseudo-color image of input hyperspectral matrix
def show_hyper_img_top(img, show_img=False):
    # Step 1:
    idx, bands = calculate_well_structured(img)
    r, b, g = calculate_rgb(idx, bands)
    pseudo_img = np.array(r, g, b)
    pseudo_img = np.transpose(pseudo_img, axes=(1, 2, 0))
    if show_img:
        cv2.imshow('Pseudo-color', pseudo_img)
        cv2.waitKey()
    print('Done.')
    return pseudo_img


if __name__ == '__main__':
    # Under Tested
    pass

