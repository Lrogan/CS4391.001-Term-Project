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
        diff_percent = .4 - aver_bright_percent
        flat_add = diff_percent * 255
        img = img + flat_add

      elif(aver_bright_percent > .6):
        diff_percent = aver_bright_percent - .6
        flat_add = diff_percent * 255
        img = img + flat_add

      BAimages.append(img)
    return BAimages

# Resize all images
def resizeImgs(imgList, size):
   imgs = []
   dim = (size, size)
   for img in imgList:
      resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
      imgs.append(resized)
   return imgs

# Extract Keypoints and Descriptors
def extractSIFT(imgList):
   

bedroomImgs, coastImgs, forestImgs = load_images_from_folder(os.getcwd())


# For each set of images adjust brightness and greate two different arrays of square size 200 and 50
bedroomImgs = brightnessAdjust(bedroomImgs)
bedroomImgs200 = resizeImgs(bedroomImgs, 200)
bedroomImgs50 = resizeImgs(bedroomImgs, 50)

coastImgs = brightnessAdjust(coastImgs)
coastImgs200 = resizeImgs(coastImgs, 200)
coastImgs50 = resizeImgs(coastImgs, 50)

forestImgs = brightnessAdjust(forestImgs)
forestImgs200 = resizeImgs(forestImgs, 200)
forestImgs50 = resizeImgs(forestImgs, 50)

# Create SIFT
sift = cv.SIFT_create()

bedroomKeypoints200, bedroomDescriptors200 = extractSIFT(bedroomImgs200)
bredroomKeypoints50, bedroomDescriptors50 = extractSIFT(bedroomImgs50)

coastKeypoints200, bedroomDescriptors200 = extractSIFT(bedroomImgs200)
bredroomKeypoints50, bedroomDescriptors50 = extractSIFT(bedroomImgs50)

bedroomKeypoints200, bedroomDescriptors200 = extractSIFT(bedroomImgs200)
bredroomKeypoints50, bedroomDescriptors50 = extractSIFT(bedroomImgs50)

  