import cv2
import numpy as np

#img = cv2.imread('inputs.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('inputs.jpg', cv2.IMREAD_COLOR)
#cv2.imshow('inputs', img)

#h, w, z = img.shape
#wNew = np.int(w >> 1)
#imgA = img[:,0 : wNew]
#imgB = img[:, wNew : w]
#
#cv2.imshow('input A', imgA)
#cv2.imshow('input B', imgB)
#
#cv2.imwrite('inputA.jpg', imgA)
#cv2.imwrite('inputB.jpg', imgB)

cpPos = []
def onMouse(event, x, y, flags, param):
    global cpLeft
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    cv2.circle(imgA, (x, y), 5, (255, 0, 0), -1)
    print('x = %d, y = %d'%(x, y))
    cpPos.append((x, y))
    cv2.imshow('InputA', imgA)

    #return x, y


imgA = cv2.imread('inputA.jpg', cv2.IMREAD_COLOR)
#imgB = cv2.imread('inputB.jpg', cv2.IMREAD_COLOUR)

cv2.namedWindow('InputA')
cv2.imshow('InputA', imgA)
#cv2.imshow('Input B', imgB)

# get the corresponding points in the left image(source image)
cv2.setMouseCallback('InputA', onMouse)

# prepare the output image
# which aspect ratio : 3 / 4
#imgOut = np.zeros((300, 400), np.uint8)

# set the corresponding points in the right image
cpRight = np.array([[0, 0], [0, 299], [399, 299], [399, 0]])
#cpRight = np.array([[0, 0], [299, 0], [299, 399], [0, 399]])
print('Right', cpRight)

cv2.waitKey(0)

#for d in cpPos:
    #print('After', d)

# get the corresponding points in the right image (source)
cpLeft = np.array(cpPos)
print('Left', cpLeft)

# Get the homography pts
h, status = cv2.findHomography(cpLeft, cpRight, cv2.RANSAC)
#h, status = cv2.findHomography(cpLeft, cpRight)
print(h)

# get the result images
imgOut = cv2.warpPerspective(imgA, h, dsize=(300, 400))

cv2.imshow('Output', imgOut)

cv2.waitKey(0)



