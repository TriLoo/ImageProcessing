import cv2
import numpy

# Step 1: read image
img = cv2.imread('imfill.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Input', img)

# Generate the new image
#width = img.shape[1]
#wNew = width / 3
#imgIn = img[:, 0 : int(wNew)]
#print('The width of input is: ', width)
#cv2.imshow('Input', img)
#cv2.imshow('After Crop', imgIn)
#print(imgIn.shape)
#cv2.imwrite('imfill.jpg', imgIn)

# Step 2: threshold the input image to obtain a binary image
imgIn = img
imgInv = numpy.zeros(img.shape, numpy.uint8)
cv2.threshold(imgIn, 200, 255, cv2.THRESH_BINARY_INV, imgInv)
cv2.imshow('Inverted', imgInv)

# Step 3: flood fill from pixel (0, 0)
h, w = img.shape[:2]
mask = numpy.zeros((h + 2, w + 2), numpy.uint8)
imgFill = imgInv.copy()
cv2.floodFill(imgFill, mask, (0, 0), 255)
#cv2.imshow('Filled Inverted', imgFill)

# Step 4: Invert the flood filled image
imgFillInv = cv2.bitwise_not(imgFill)

# Step 5: combine flood filled image with threshold input image with OR
imgOut = numpy.zeros(img.shape, numpy.uint8)
cv2.bitwise_or(imgInv, imgFillInv, imgOut)

cv2.imshow('Result', imgOut)

cv2.waitKey(0)

