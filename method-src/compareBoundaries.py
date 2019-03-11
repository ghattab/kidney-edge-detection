import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import spatial
import re
import random
from datetime import datetime
from reverseDistField import convertImage
import os

generatedBoundariesPath = "testSetPredictions/4658635_run12/boundary4658635"
groundTruthPath = "Data/kidney_dataset_1_15_24GB"

def getImagePair(generatedBoundariesPath, groundTruthPath):
	subdirs = [x[0] for x in os.walk(generatedBoundariesPath)]

	subdirs = sorted(subdirs)

	for subdir in subdirs[1:]:
		subdirNumber = re.findall(r'\d+', subdir)[-1]
		generatedImages = [f for f in listdir(subdir) if isfile(join(subdir, f))]

		gtPathPrefix = "kidney_dataset_" + str(subdirNumber)
		# Match generated images with ground thruth Images 
		# Image names need to match for this

		for image in generatedImages:
			generated = cv2.imread(subdir + "/" + image, 0)
			print(join(groundTruthPath, gtPathPrefix) + "/ground_truth/" + image)
			groundTruth = cv2.imread(join(groundTruthPath, gtPathPrefix) + "/ground_truth/" + image, 0)
			groundTruth = groundTruth[28:28+1024, 320:320+1280]

			assert (generated.shape == groundTruth.shape), "{}, {}".format(generated.shape, groundTruth.shape)

			yield ("x" + subdirNumber + "/" + image, generated, groundTruth)

def getContourPoints(image):
	result = []

	it = np.nditer(image, flags=["multi_index"])
	while not it.finished:
		if it[0] == 255 or it[0] == 1.0:
			result.append(it.multi_index)
		it.iternext()

	return np.array(result)

def score(basePoints, comparisonPoints):
	if len(comparisonPoints) == 0:
		return len(basePoints) * 1640

	comparisonPointTree = spatial.KDTree(comparisonPoints)

	totalDistance = 0.0
	for point in basePoints:
		totalDistance += comparisonPointTree.query(point)[0]

	return totalDistance

def compairImagePair(generatedImage, groundTruth):
	generatedContourPoints = getContourPoints(generatedImage)
	groundThruthContourPoints = getContourPoints(groundTruth)
	
	precisionScore = score(groundThruthContourPoints, generatedContourPoints)
	recallScore = score(generatedContourPoints, groundThruthContourPoints)

	return (precisionScore, recallScore)

if __name__ == "__main__":
	random.seed(datetime.now())
	with open("comparisonResult{}.csv".format(int(random.random() * (2 ** 20))), "w") as out:
		out.write("Path, Precision, Recall\n")
		for (namePath, generatedImage, gtImage) in getImagePair(generatedBoundariesPath, groundTruthPath):
			precision, recall = compairImagePair(generatedImage, gtImage)
			out.write("{}, {}, {}\n".format(namePath, precision, recall))
