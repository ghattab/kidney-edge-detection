import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from skimage import morphology
from thinning import guo_hall_thinning
import re
from multiprocessing import Pool

# absoluteThreshold = 40
absoluteThreshold = 120

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

	image[image < absoluteThreshold] = 0
	image[image < avg] = 0
	
	image[image >= avg] = 1
	
	for _ in range(10):
		image = morphology.binary_erosion(image)

	# image = np.multiply(image.astype(np.ubyte), 255)
	# image = guo_hall_thinning(image)

	image = morphology.skeletonize(image)
	result = np.multiply(image, 255.0)
	# result = result.astype(np.uint8)
	return result

def convertImages(paths):
	srcPath, dstPath = paths
	if not os.path.exists(dstPath):
		os.makedirs(dstPath)

	for name, image in yieldImages(srcPath):
		print("{}".format(name), end="\r")

		out = convertImage(image)

		os.makedirs(os.path.dirname(dstPath + "/" + name), exist_ok=True)
		cv2.imwrite(dstPath + "/" + name, out)

def main():
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

	dstPaths = ["crossValidationPredictions/boundaries_original_algo/strictVersion/boundary_wo1/test",]
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo2/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo3/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo4/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo5/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo6/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo7/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo8/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo9/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo10/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo11/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo12/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo13/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo14/test",
				# "crossValidationPredictions/boundaries_original_algo/boundary_wo15/test"]

	p = Pool()
	paths = zip(srcPaths, dstPaths)
	p.map(convertImages, paths)

if __name__ == "__main__":
	main()