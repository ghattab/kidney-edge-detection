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

# parser = argparse.ArgumentParser(description='Calculate disp map.')
# parser.add_argument('-d', metavar='dataPath', type=str,
#                     help='Data path')
# args = parser.parse_args()

# dataPath = "D:/KidneyChallenge/kidney_1_4_training/kidney_dataset_1/"
dataPaths = [	
				"Data/kidney_dataset_5/",
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
leftFolder = "croppedLeft_frames/"
disparityDirectory = "disparity/"

# Destination folder
destination = "x"

imageSize = (1280, 1024)

def dirCount(directory):
	return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def getImagePair(dataPath):
	global imageSize
	global leftFolder
	global disparityDirectory

	leftCount = dirCount(dataPath + leftFolder)
	dispCount = dirCount(dataPath + disparityDirectory)

	if leftCount != dispCount:
		raise ValueError("Left and Right Image folders do not have matching image count.")

	leftFileNames = [f for f in listdir(dataPath + leftFolder) if isfile(join(dataPath + leftFolder, f))]

	for imageName in leftFileNames:
		left = cv2.imread(dataPath + leftFolder + imageName)
		disp = cv2.imread(dataPath + disparityDirectory + imageName, 0)

		yield left, disp, imageName

def combineImages(dataPath, counter):
	for leftImage, disp, imageName in getImagePair(dataPath):
		exDisp = np.expand_dims(disp, axis=2)
		combined = np.concatenate((leftImage, exDisp), axis=2)
		
		cv2.imwrite(dataPath + destination + str(counter) + "/" + imageName, combined, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == "__main__":
	counter = 4
	for dataPath in dataPaths:
		counter += 1
		if not os.path.exists(dataPath + destination):
			os.makedirs(dataPath + destination + str(counter))

		print("Combining: {}...".format(dataPath))
		
		combineImages(dataPath, counter)