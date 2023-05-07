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
    label = []
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

          if 'bedroom' in filePath.lower():
            label.append(0)
          elif 'coast' in filePath.lower():
            label.append(1)
          elif 'forest' in filePath.lower():
            label.append(2)
          
    return imgs50, imgs200, descriptors50, descriptors200, label

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

  # homogenize shape of descriptors so that its always (128, X where X >= 128)
  if desc.shape[0] < 128:
      desc = np.vstack((desc, np.zeros((128 - desc.shape[0], 128))))
  if desc.shape[0] > 128:
      desc = desc[:128, :]

  desc = np.float32(desc)
  desc = np.array(desc.flatten())
  return desc

def calcPercents(res, y):
  return

# Load and process training images including SIFT descriptors
print("Loading Training Images.")
imgs50, imgs200, descriptors50, descriptors200, labels = load_images_from_folder(os.path.join(os.getcwd(), 'ProjData', 'Train'))
imgs50 = np.array(imgs50)
imgs200 = np.array(imgs200)
descriptors50 = np.array(descriptors50)
descriptors200 = np.array(descriptors200)
labels = np.array(labels)

# Load and consolidate test images including SIFT descriptors
print("Loading Test Images.")
test50, test200, testDescriptors50, testDescriptors200, testLabels = load_images_from_folder(os.path.join(os.getcwd(), 'ProjData', 'Test'))
test50 = np.array(test50)
test200 = np.array(test200)
testDescriptors50 = np.array(testDescriptors50)
testDescriptors200 = np.array(testDescriptors200)
testLabels = np.array(testLabels)


# KNN Raw
print("KNN Raw")
knnRaw = cv.ml.KNearest_create()
XRaw = np.array([i.flatten() for i in imgs50], dtype=np.float32)
y = labels
knnRaw.train(XRaw, cv.ml.ROW_SAMPLE, y)

XRawTest = np.array([i.flatten() for i in test50], dtype=np.float32)
_, resRaw, _, _ = knnRaw.findNearest(XRawTest,k=1)
calcPercents(resRaw, y)


# KNN SIFT
# train_labels = None


# # KNN SVM
