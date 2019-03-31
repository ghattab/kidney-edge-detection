import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from skimage import morphology, measure, filters
from scipy.ndimage.morphology import binary_hit_or_miss, binary_fill_holes
import re
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys

# absoluteThreshold = 40
absoluteThreshold = 120

numIterations = 60

def connectedComponents(image):
	binaryImage = np.copy(image)
	binaryImage[binaryImage > 0] = 1

	labeledImage = measure.label(binaryImage, background=0)

	labels = np.unique(labeledImage)
	for label in labels[1:]:
		partialImage = np.copy(labeledImage)
		partialImage[partialImage != label] = 0
		partialImage[partialImage > 0] = 1
		yield np.logical_and(partialImage, image).astype(np.uint8)

def filterSmallConnectedComponents(image, threshold=100):
	imageMask = np.copy(image)

	for component in connectedComponents(imageMask):
		count = np.count_nonzero(component)
		if count < threshold:
			imageMask = np.logical_xor(imageMask, component)

	return np.logical_and(image, imageMask).astype(np.uint8)

def yieldImages(path):
	subdirs = [x[0] for x in os.walk(path)]

	for subdir in subdirs[1:]:
		subdirNumber = re.findall(r'\d+', subdir)[-1]
	
		fileNames = [f for f in listdir(subdir) if isfile(join(subdir, f))]
		fileNames = sorted(fileNames)

		for file in fileNames:
			distFieldImage = cv2.imread(join(subdir, file), 0)

			yield ("x" + str(subdirNumber) + "/" + file, distFieldImage)

def removeSpurs(image, numIterations=60):
	kernels = np.array([
		[[-1, 0, 0],
		 [-1, 1, -1],
		 [-1, -1, -1]],
		 [[0, 0, -1],
		 [-1, 1, -1],
		 [-1, -1, -1]],
		 ])
	for _ in range(numIterations):
		for kernel in kernels:
			for x in range(4):
				spurs = cv2.morphologyEx(image.astype(np.uint8), kernel=np.rot90(kernel, k=x), op=cv2.MORPH_HITMISS, borderValue=0)
				image = np.logical_xor(image, spurs)
	return image

def reconstructEndpoints(image, nonPrunedImage, numIterations=40):
	kernels = np.array([
		[[-1, 0, 0],
		 [-1, 1, -1],
		 [-1, -1, -1]],
		 [[0, 0, -1],
		 [-1, 1, -1],
		 [-1, -1, -1]],
		 ])
	dilateKernel = np.array([[1, 1, 1],
		 					[1, 1, 1],
		 					[1, 1, 1]])

	endpoints = np.zeros(image.shape)

	image = image.astype(np.uint8)

	# Find endpoints
	for kernel in kernels:
		for x in range(4):
			ends = cv2.morphologyEx(image, kernel=np.rot90(kernel, k=x), op=cv2.MORPH_HITMISS, borderValue=0)
			endpoints = np.logical_or(endpoints, ends)

	# Conditional dilation
	for _ in range(numIterations):
		dilated = cv2.dilate(endpoints.astype(np.uint8), kernel=dilateKernel)
		endpoints = np.logical_and(nonPrunedImage, dilated)

	return np.logical_or(image, endpoints)

def convertImage(image):
	image = cv2.resize(image, (1280, 1024))
	avg = np.average(image) * 0.9
	
	image[image < absoluteThreshold] = 0
	image[image < avg] = 0
	
	image[image >= avg] = 1

	image = filterSmallConnectedComponents(image)
	
	for _ in range(10):
		image = morphology.binary_erosion(image)

	image = morphology.skeletonize(image)
	image = binary_fill_holes(image)
	image = morphology.skeletonize(image)

	nonPrunedImage = image
	image = removeSpurs(image)
	image = reconstructEndpoints(image, nonPrunedImage)

	result = np.multiply(image, 255.0)
	return result

def convertImages(paths):
	srcPath, dstPath = paths
	if not os.path.exists(dstPath):
		os.makedirs(dstPath)

	print(srcPath)
	for name, image in yieldImages(srcPath):
		out = convertImage(image)

		os.makedirs(os.path.dirname(dstPath + "/" + name), exist_ok=True)
		cv2.imwrite(dstPath + "/" + name, out)

def main(r_lower=1, r_upper=15, prefix=""):
	srcPaths = ["crossValidationPredictions/pred/pred_wo1/test",]
				# "crossValidationPredictions/pred/pred_wo2/test",
				# "crossValidationPredictions/pred/pred_wo3/test",
				# "crossValidationPredictions/pred/pred_wo4/test",
				# "crossValidationPredictions/pred/pred_wo5/test",
				# "crossValidationPredictions/pred/pred_wo6/test",
				# "crossValidationPredictions/pred/pred_wo7/test",
				# "crossValidationPredictions/pred/pred_wo8/test",
				# "crossValidationPredictions/pred/pred_wo9/test",
				# "crossValidationPredictions/pred/pred_wo10/test",
				# "crossValidationPredictions/pred/pred_wo11/test",
				# "crossValidationPredictions/pred/pred_wo12/test",
				# "crossValidationPredictions/pred/pred_wo13/test",
				# "crossValidationPredictions/pred/pred_wo14/test",
				# "crossValidationPredictions/pred/pred_wo15/test"]

	dstPaths = ["crossValidationPredictions/boundaries_original_algo_spur_removal/strictVersion/boundary_strict_wo1/test",]
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo2/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo3/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo4/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo5/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo6/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo7/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo8/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo9/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo10/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo11/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo12/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo13/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo14/test",
				# "crossValidationPredictions/boundaries_original_algo_spur_removal/boundary_strict_wo15/test"]
	srcPaths = srcPaths[r_lower-1:r_upper]
	dstPaths = dstPaths[r_lower-1:r_upper]

	if prefix:
		srcPaths = [prefix + x for x in srcPaths]
		dstPaths = [prefix + x for x in dstPaths]

	# p = Pool()
	paths = zip(srcPaths, dstPaths)
	# p.map(convertImages, paths)

	convertImages(next(paths))

if __name__ == "__main__":
	if len(sys.argv) == 4:
		r_lower = int(sys.argv[1])
		r_upper = int(sys.argv[2])
		prefix = sys.argv[3]
		main(r_lower, r_upper, prefix)
	elif len(sys.argv) == 2:
		main(prefix=sys.argv[1])
	else:
		main()