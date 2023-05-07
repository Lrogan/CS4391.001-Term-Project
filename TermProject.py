import math
import os
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# read image and convert to grayscale
def readImg(path):
  img = cv.imread(path, cv.IMREAD_GRAYSCALE)
  if img is not None:
      return img
  else:
      print(path + " failed to read.")
      exit()

# Read all test images
def load_images_from_folder(folder):
    currPath = folder
    imgs50 = []
    imgs200 = []
    descriptors50 = []
    descriptors200 = []
    for filename in os.listdir(currPath):
        currPath = os.path.join(folder, filename)
        for file in os.listdir(currPath):
          
          filePath = os.path.join(currPath, file)
          img = readImg(filePath)
          img = brightnessAdjust(img)
          img50 = resizeImg(img, 50)
          img200 = resizeImg(img, 200)
          desc50 = extractSIFT(img50)
          desc200 = extractSIFT(img200)

          imgs50.append(img50)
          imgs200.append(img200)
          descriptors50.append(desc50)
          descriptors200.append(desc200)

    return imgs50, imgs200, descriptors50, descriptors200

# Determine adjust brightness according to avg brightness
def brightnessAdjust(img):
  brightness = np.mean(img) / 255
  bImg = img
  if brightness < .4:
      bImg = cv.addWeighted(img, 1.5, 0 ,0 ,0)
  elif brightness > .6:
      bImg = cv.addWeighted(img, .5, 0, 0, 0)
  return bImg

# Resize all images
def resizeImg(img, size):
  dim = (size, size)
  resized = cv.resize(img, dim)
  return resized

# Extract Keypoints and Descriptors
def extractSIFT(img):
  sift = cv.SIFT_create()
  kp, desc = sift.detectAndCompute(img, None)

  # if no keypoints, set to 0
  if desc is None:
      desc = np.zeros((1, 128))

  # homogenize shape of descriptors
  if desc.shape[0] < 128:
      desc = np.vstack((desc, np.zeros((128 - desc.shape[0], 128))))
  if desc.shape[0] > 128:
      desc = desc[:128, :]

  desc = np.float32(desc)
  desc = np.array(desc.flatten())
  return desc

# Load and process training images including SIFT descriptors
imgs50, imgs200, descriptors50, descriptors200 = load_images_from_folder(os.path.join(os.getcwd(), 'ProjData', 'Train'))
imgs50 = np.array(imgs50)
imgs200 = np.array(imgs200)
descriptors50 = np.array(descriptors50)
descriptors200 = np.array(descriptors200)

# Load and consolidate test images
test50, test200, testDescriptors50, testDescriptors200 = load_images_from_folder(os.path.join(os.getcwd(), 'ProjData', 'Test'))

X = 0

# KNN Raw
knn = cv.ml.KNearest_create()
k = np.arange(10)
train_labels = np.repeat(k, 250)

# knn.train(imgs50, cv.ml.ROW_SAMPLE, train_labels)
# ret,result,neighbours,dist = knn.findNearest(imgs0,k=5)
# print(str(ret))
# print(str(result))
# print(str(neighbours))
# print(str(dist))


# KNN SIFT
# train_labels = None


# # KNN SVM
