import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from skimage import morphology
from thinning import guo_hall_thinning
import re

srcPath = "testSetPredictions/4658635_run12/pred4658635"
dstPath = "testSetPredictions/4658635_run12/boundary4658635"

absoluteThreshold = 40

def yieldImages(path):
	subdirs = [x[0] for x in os.walk(path)]

	for subdir in subdirs[1:]:
		subdirNumber = re.findall(r'\d+', subdir)[-1]
	
		fileNames = [f for f in listdir(subdir) if isfile(join(subdir, f))]
		fileNames = sorted(fileNames)

		for file in fileNames:
			distFieldImage = cv2.imread(join(subdir, file), cv2.IMREAD_UNCHANGED)

			yield ("x" + str(subdirNumber) + "/" + file, distFieldImage)

def convertImage(image):
	image = cv2.resize(image, (1280, 1024))
	avg = np.average(image) * 0.9
	image[image < avg] = 0
	image[image < absoluteThreshold] = 0
	image[image >= avg] = 1
	
	for _ in range(10):
		image = morphology.binary_erosion(image)

	# image = np.multiply(image.astype(np.ubyte), 255)
	# image = guo_hall_thinning(image)

	image = morphology.skeletonize_3d(image)
	result = np.multiply(image, 255.0)
	# result = result.astype(np.uint8)
	return result

def convertImages(srcPath, dstPath):
	if not os.path.exists(dstPath):
		os.makedirs(dstPath)

	for name, image in yieldImages(srcPath):
		print("{}".format(name), end="\r")

		out = convertImage(image)

		os.makedirs(os.path.dirname(dstPath + "/" + name), exist_ok=True)
		cv2.imwrite(dstPath + "/" + name, out, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == "__main__":
	convertImages(srcPath, dstPath)