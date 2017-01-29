import numpy as np
import os
import cv2 as cv


os.chdir('train/')

HEIGHT = 224
WIDTH= 224

for imgFile in os.listdir('.'):
    image = cv.imread(imgFile)
    flipped = image[:,:,::-1]
    cv.imwrite(flipped, imgFile + '_reflect')