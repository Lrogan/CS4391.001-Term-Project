import math
import os
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Convert rbg to HSV

# Determine adjust brightness according to avg brightness
def brightnessAdjust(imglist):
    BAimages = []
    for img in imglist:
      value = np.sum(img[::])
      pxl = img.shape[0] * img.shape[1]
      aver_bright = value / pxl
      if(aver_bright < .4):
        return 
      elif(aver_bright > .6):
        return 
      BAimages.append(img)
    return BAimages

# Read all test images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if img is not None:
            images.append(img)
        else:
            print(filename + " failed to read.")
    return images

imglist = load_images_from_folder()
imglist = brightnessAdjust(imglist)