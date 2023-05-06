import math
import os
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# read image and convert to grayscale
def readImg(path):
  img = cv.imread(path)
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  if img is not None:
      return img
  else:
      print(path + " failed to read.")
      exit()

# Read all test images
def load_images_from_folder(folder):
    currPath = folder
    images = [[],[],[]]
    for filename in os.listdir(currPath):
        currPath = os.path.join(folder, filename)
        if os.path.isdir(filename):
          for file in os.listdir(currPath):
            currPath = os.path.join(currPath, file)
            if 'bedroom' in currPath:
              images[0].append(readImg(currPath))
            if 'coast' in currPath:
              images[1].append(readImg(currPath))
            if 'forest' in currPath:
              images[2].append(readImg(currPath))
            
    return images

# Determine adjust brightness according to avg brightness
def brightnessAdjust(imglist):
    BAimages = []
    for img in imglist:
      value = np.sum(img[::])
      pxl = img.shape[0] * img.shape[1]
      aver_bright = value / pxl
      aver_bright_percent = (aver_bright / 255)
      if(aver_bright_percent < .4):
        return 
      elif(aver_bright_percent > .6):
        return 
      BAimages.append(img)
    return BAimages

# Resize all images
def resizeImgs(imgList, size):
   imgs = []
   dim = (size, size)
   for img in imgList:
      resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
      imgs.append(resized)


bedroomImgs, coatImgs, forestImgs = load_images_from_folder(os.getcwd())

bedroomImgs = brightnessAdjust(bedroomImgs)
bedroomImgs200 = resizeImgs(bedroomImgs, 200)
bedroomImgs50 = resizeImgs(bedroomImgs, 50)

coatImgs = brightnessAdjust(coatImgs)
coatImgs = resizeImgs(coatImgs, 200)
coatImgs = resizeImgs(coatImgs, 50)

forestImgs = brightnessAdjust(forestImgs)
forestImgs = resizeImgs(forestImgs, 200)
forestImgs = resizeImgs(forestImgs, 50)
