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
    print(os.listdir(currPath))
    for filename in os.listdir(currPath):
        
        currPath = os.path.join(folder, filename)
        print(currPath)
        for file in os.listdir(currPath):
          
          filePath = os.path.join(currPath, file)

          if 'bedroom' in filePath.lower():
            images[0].append(readImg(filePath))

          if 'coast' in filePath.lower():
            images[1].append(readImg(filePath))

          if 'forest' in filePath.lower():
            images[2].append(readImg(filePath))
    return images[0], images[1], images[2], images[0].append(images[1].append(images[2]))

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
        flat_add = math.ceil(diff_percent * 255)
        img = img + flat_add

      elif(aver_bright_percent > .6):
        diff_percent = aver_bright_percent - .6
        flat_add = math.ceil(diff_percent * 255)
        img = img - flat_add

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
  sift = cv.SIFT_create()
  desc = np.empty(0, dtype=np.uint32)
  kp = np.empty(0, dtype=np.uint32)

  for img in imgList:
    k, d = sift.detectAndCompute(img, None)
    desc = np.append(desc, d)
    kp = np.append(kp, k)
  
  return kp, desc


bedroomImgs, coastImgs, forestImgs, allImgs = load_images_from_folder(os.path.join(os.getcwd(), 'ProjData', 'Train'))
print(len(bedroomImgs), len(coastImgs), len(forestImgs))

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

# Consolidate Images
imgs200 = np.empty(0, dtype=np.uint32)
imgs200 = np.append(imgs200, bedroomImgs200)
imgs200 = np.append(imgs200, coastImgs200)
imgs200 = np.append(imgs200, forestImgs200)
print("Shape of imgs200: " + str(imgs200.shape))

imgs50 = np.empty(0, dtype=np.uint32)
imgs50 = np.append(imgs50, bedroomImgs50)
imgs50 = np.append(imgs50, coastImgs50)
imgs50 = np.append(imgs50, forestImgs50)
print("Shape of imgs50: " + str(imgs50.shape))

# Create and save SIFT features
bedroomKeypoints200, bedroomDescriptors200 = extractSIFT(bedroomImgs200)
bedroomKeypoints50, bedroomDescriptors50 = extractSIFT(bedroomImgs50)

coastKeypoints200, coastDescriptors200 = extractSIFT(bedroomImgs200)
coastKeypoints50, coastDescriptors50 = extractSIFT(bedroomImgs50)

forestKeypoints200, forestDescriptors200 = extractSIFT(forestImgs200)
forestKeypoints50, forestDescriptors50 = extractSIFT(forestImgs50)

# Consolidate Keypoints of 200 and 50 sizes separately
keypoints200 = np.empty(0, dtype=np.uint32)
keypoints200 = np.append(keypoints200, bedroomKeypoints200)
keypoints200 = np.append(keypoints200, coastKeypoints200)
keypoints200 = np.append(keypoints200, forestDescriptors200)

keypoints50 = np.empty(0, dtype=np.uint32)
keypoints50 = np.append(keypoints50, bedroomKeypoints50)
keypoints50 = np.append(keypoints50, coastKeypoints50)
keypoints50 = np.append(keypoints50, forestKeypoints50)

# Consolidate Descriptors of 200 and 50 sizes separately
descriptors200 = np.empty(0, dtype=np.uint32)
descriptors200 = np.append(descriptors200, bedroomDescriptors200)
descriptors200 = np.append(descriptors200, coastDescriptors200)
descriptors200 = np.append(descriptors200, forestDescriptors200)

descriptors50 = np.empty(0, dtype=np.uint32)
descriptors50 = np.append(descriptors50, bedroomDescriptors50)
descriptors50 = np.append(descriptors50, coastDescriptors50)
descriptors50 = np.append(descriptors50, forestDescriptors50)

descriptors200 = descriptors200.flatten()
descriptors50 = descriptors50.flatten()

# print("Descriptors\n")
# print(descriptors200.size)
# print(str(descriptors200))
# print("\n\n")
# print(descriptors50.size)
# print(str(descriptors50))

# Load and consolidate test images
test = np.empty(0, dtype=np.uint32)
imgs0, imgs1, imgs2, imgsA = load_images_from_folder(os.path.join(os.getcwd(), 'ProjData', 'Test'))
print(len(imgs0), len(imgs1), len(imgs2))
test = np.append(test, np.array(imgs0))
test = np.append(test, np.array(imgs1))
test = np.append(test, np.array(imgs2))


# # KNN Raw
# knn = cv.ml.KNearest_create()
# k = np.arange(10)
# train_labels = np.repeat(k, 250)

# knn.train(imgs50, cv.ml.ROW_SAMPLE, train_labels)
# ret,result,neighbours,dist = knn.findNearest(imgs0,k=5)
# print(str(ret))
# print(str(result))
# print(str(neighbours))
# print(str(dist))


# KNN SIFT
# train_labels = None


# # KNN SVM
