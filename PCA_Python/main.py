#encoding=UTF-8
#import cv2
from __future__ import print_function
from PIL import Image
import numpy as np
import os, platform
#from matplotlib import pyplot as plt

def ImageRead(path, name):
    '''
    Read a image.

    :param path:
    :param name:
    :return:
    '''
    try:
        img = Image.open(os.path.join(path, name)).convert('L')
    except IOError:
        print('Read Image failed ...')
    else:
        print(img.format, img.size, img.mode)
    return img

class myPCA:
    '''
    This is a class about PCA based on Python.
    '''
    def __init__(self, pre = 0.90):
        self.__precison = pre

    def __zeroMean(self, dataMat):
        means = np.mean(dataMat, axis=0)           # the average value of each feature, axis = 0 ==> per column
        print(means.size)            # for test
        # return a 2 * 1 tuple
        return dataMat - means, means              # including a hidden broadcast operation

    def __percentage2n(self, eigVals):
        sortArray = np.sort(eigVals)              # 升序
        sortArray = sortArray[-1::-1]
        arraySum = np.sum(sortArray)
        tempSum = 0
        num = 0;
        for i in sortArray:
            tempSum += i
            num += 1
            if tempSum >= arraySum * self.__precison:
                break
        return num

    def pca(self, dataIn):
        newData, meanVals = self.__zeroMean(dataIn)
        print(newData.shape)     # for test
        covMat = np.cov(newData, rowvar=0)
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))    # calculate the covariance matrix
        n = self.__percentage2n(eigVals)             # get the 'n'
        # For Test
        print('n = {}'.format(n))
        index_array = np.argsort(eigVals)
        n_eigValIndex = index_array[-1:-(n+1):-1]    # 从后面数 n 个下标
        n_eigVects = eigVects[:, n_eigValIndex]      # 取出对应的 n 个列向量
        lowDDataMat = newData * n_eigVects
        reconMat = (lowDDataMat * n_eigVects.T) + meanVals
        '''
        index_array = np.argsort(eigVals)
        print(index_array)     # for test
        new_eigVals = eigVals[index_array]           # get the sorted array
        totalEigSum = np.sum(new_eigVals)
        tempSum = 0
        n = 0
        for i in range(new_eigVals.size):
            tempSum += new_eigVals[:-i]
            n += 1
            if tempSum / totalEigSum > self.precison:
                break
        new_eigVects = eigVects[]
        '''
        return lowDDataMat, reconMat

# obtain the grayscale image
img = ImageRead('./', 'lena.jpg')

# For test
#img.show()

# change the Image class data to numpy matrix class
imgMat = np.array(img)
'''
imgMat = np.matrix(img.getdata(), dtype='float')
imgMat = imgMat.reshape(img.size[0], img.size[1])
'''

mp = myPCA(0.95)

decMat, resMat = mp.pca(imgMat)

decImg = Image.fromarray(decMat)
recImg = Image.fromarray(resMat)

decImg.show('Decompose')
recImg.show('Restore')

