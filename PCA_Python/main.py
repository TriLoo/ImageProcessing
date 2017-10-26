#encoding=UTF-8


#from numpy import * as np    # only import some functions and args that not begin with underline('_')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def zeroMean(dataMat):
    '''
    Get the zero mean matrix
    :param dataMat:
    :return:
    '''
    meanVal = np.mean(dataMat, axis=0)   # one feature per column, get a row vector
    print(meanVal.size)
    newData = dataMat - meanVal          # Broadcast
    return newData, meanVal

def pca(dataMat, n):
    '''
    Top PCA function.
    :param dataMat:
    :param n:
    :return:
    '''
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)      # rowvar : one sample per line, return ndarray

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))        # calculate the eig-vectors and eig-values
    eigValIndice = np.argsort(eigVals)                       # 对特征值排序，从小到大
    n_eigValIndice = eigValIndice[-1:-(n+1):-1]              # 最大的n个特征值的下标

    n_eigVect = eigVects[:, n_eigValIndice]
    lowDDataMat = newData * n_eigVect                        # 低维特征空间的数据
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal
    return lowDDataMat, reconMat

# Read data
im = Image.open('lena.jpg')
im_array = np.array(im)
print(im_array.size)           # = row * col * channels
# 显示图像
im.show()
#convert to gray scale
im_gray = im.convert("L")
im_gray.show()

lowDData, reconMat = pca(im_gray, 20)

plt.axis('off')
plt.figure("Low")
plt.imshow(lowDData)
plt.show()
plt.figure("Rec")
plt.imshow(reconMat)
plt.show()

