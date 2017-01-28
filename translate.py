import cv2 as cv
import os
from skimage.measure import block_reduce
import numpy as np


os.chdir('train/')

HEIGHT = 224
WIDTH = 224

for imgFile in os.listdir('.'):
    if imgFile.endswith('.jpg'):
        image = cv.imread(imgFile)
        newImg = block_reduce(image, block_size = (3,3,1), func = np.mean)
        

        # there are no picture that are less than 224 after the downsampling 
        if shape[1] < WIDTH: 
            newImg = adjustWidth(newImg, shape[1], shape[0])
            shape = newImg.shape

        diffHeight = shape[0] - HEIGHT
        diffWidth = shape[1] - WIDTH

        for i in range(diffWidth):
            for j in range(diffHeight):
                imgCreate = newImg[j : HEIGHT + j, i : WIDTH + i, :]
                newFilename = imgFile + '_' + str(i + j)
                cv.imwrite(newFilename, imgCreate)



def adjustWidth(img, imgWidth, imgHeight):
    diff = WIDTH - imgWidth
    # pad with black 
    newImg = np.zeros((imgHeight, WIDTH, 3))
    
    newImg = img[:, : imgWidth, :]
    return newImg
