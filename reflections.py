import numpy as np
import os
import cv2 as cv


os.chdir('train/')

HEIGHT = 224
WIDTH= 224

for imgFile in os.listdir('.'):
    flipped = cv.imread(imgFile)
    
