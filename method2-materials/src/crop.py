import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import math
import argparse
from scipy.interpolate import splprep, splev


# parser = argparse.ArgumentParser(description='Calculate disp map.')
# parser.add_argument('-d', metavar='dataPath', type=str,
#                     help='Data path')
# args = parser.parse_args()

# dataPath = "D:/KidneyChallenge/kidney_1_4_training/kidney_dataset_1/"
dataPaths = [	"Data/kidney_dataset_5/",
				"Data/kidney_dataset_6/",
				"Data/kidney_dataset_7/",
				"Data/kidney_dataset_8/",
				"Data/kidney_dataset_9/",
				"Data/kidney_dataset_10/",
				"Data/kidney_dataset_11/",
				"Data/kidney_dataset_12/",
				"Data/kidney_dataset_13/",
				"Data/kidney_dataset_14/",
				"Data/kidney_dataset_15/"
]
# dataPath = args.d
rightFolder = "right_frames/"
leftFolder = "left_frames/"

imageSize = (1280, 1024)

def dirCount(directory):
	return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def getNextImage(path):
	fileNames = [f for f in listdir(path) if isfile(join(path, f))]

	for imageName in fileNames:
		image = cv2.imread(path + imageName)

		yield image, imageName

def cropImages(loadPath, savePath):
	for image, imageName in getNextImage(loadPath):
		width, height = imageSize

		croppedImage = image[28:28+height, 320:320+width]
		cv2.imwrite(savePath + imageName, croppedImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == "__main__":
	for dataPath in dataPaths:
		if not os.path.exists(dataPath + "croppedLeft_frames/"):
			os.makedirs(dataPath + "croppedLeft_frames/")

		print("Cropping: {}...".format(dataPath))

		cropImages(dataPath + leftFolder, dataPath + "croppedLeft_frames/")