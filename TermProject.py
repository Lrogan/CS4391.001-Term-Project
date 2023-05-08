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

def calcPercents(res, y, classy):
  results = {
    "TP": 0,
    "FP": 0,
    "FN": 0
  }
  for classifier in range(3):
    total = 0
    truePos = 0
    falsePos = 0
    falseNeg = 0
    for index, label in enumerate(y):
        if label == classifier:
            total += 1
            if res[index] == label:
                truePos += 1
            else:
                falseNeg += 1
        elif res[index] == classifier:
            total += 1
            falsePos += 1
    truePos /= total
    falsePos /= total
    falseNeg /= total
    results["TP"] = truePos
    results["FP"] = falsePos
    results["FN"] = falseNeg
  
  print(f"{classy} Results In Percentage")
  print("True Positive: {:.2f}%\nFalse Positive: {:.2f}%\nFalse Negative: {:.2f}%".format(results['TP'] * 100, results['FP'] * 100, results['FN'] * 100))

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
knnSIFT = cv.ml.KNearest_create()
XRaw = np.array([i.flatten() for i in imgs50], dtype=np.float32)
yRaw = labels
knnSIFT.train(XRaw, cv.ml.ROW_SAMPLE, yRaw)

# Don't have to do this due to instructions saying only 2 classifiers needed to be tested, but the training still needs to be done
# XRawTest = np.array([i.flatten() for i in test50], dtype=np.float32)
# _, resRaw, _, _ = knnRaw.findNearest(XRawTest,k=1)
# calcPercents(resRaw, y)

# KNN SIFT
knnSIFT = cv.ml.KNearest_create()

# Creates a list consisting of two lists alternated
XSIFT = np.array([val for pair in zip(descriptors50, descriptors200) for val in pair]).astype('float32')
ySIFT = np.array([val for pair in zip(labels, labels) for val in pair]).astype('float32')
knnSIFT.train(XSIFT, cv.ml.ROW_SAMPLE, ySIFT)

XSIFTTest = np.array([val for pair in zip(testDescriptors50, testDescriptors200) for val in pair]).astype('float32')
ySIFTTest = np.array([val for pair in zip(testLabels, testLabels) for val in pair]).astype('float32')
_, resSIFT, _, _ = knnSIFT.findNearest(XSIFTTest,k=1)
print("\n")
calcPercents(resSIFT, ySIFTTest, "KNN SIFT")

# SVM
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

XSVM = np.array([val for pair in zip(descriptors50, descriptors200) for val in pair])
ySVM = np.array([val for pair in zip(labels, labels) for val in pair])
svm.train(XSVM, cv.ml.ROW_SAMPLE, ySVM)

XSVMTest = np.array([val for pair in zip(testDescriptors50, testDescriptors200) for val in pair])
ySVMTest = np.array([val for pair in zip(testLabels, testLabels) for val in pair])
resSVM = svm.predict(XSVMTest)[1]
print("\n")
calcPercents(resSVM, ySVMTest, "SVM")
