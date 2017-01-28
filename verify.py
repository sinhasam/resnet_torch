import os
import cv2 as cv

os.chdir('train')

countHeight = 0
countWidth = 0
totalCount = 0

HEIGHT = 224*3
WIDTH = 224*3

for imgFile in os.listdir('.'):
    if imgFile.endswith('.jpg'):
        totalCount += 1
        filename = str(imgFile)
        img = cv.imread(filename)
        print(filename)
        shape = img.shape

        if shape[0] > HEIGHT:
            countHeight += 1

        if shape[1] > WIDTH:
            countWidth += 1


print('width: ', countWidth)
print('height: ', countHeight)
print('total: ', totalCount)