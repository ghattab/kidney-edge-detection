import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from skimage import morphology
from thinning import guo_hall_thinning
import re
from multiprocessing import Pool

""" Script to convert the networks output image to a 1 px boundary image.
	See source and dst paths in main() and the parameter absoluteThreshold below.
"""

##### Controls the strictness: 40 -> non-strict; 120 -> strict ###
absoluteThreshold = 40		# strict
# absoluteThreshold = 120	# non-strict
##################################################################

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
	""" Converts an image to a 1 px boundary image.
		Step 1: Resize to 1280x1024
		Step 2: Thresholding 
		Step 3: Set all remaining pixels to 1 
		Step 4: 10 times binary erosion
		Step 5: 3D skeletonization
		Step 6: Set all boundary pixels to 255 
	""" 
	image = cv2.resize(image, (1280, 1024))
	avg = np.average(image) * 0.9
	
	image[image < absoluteThreshold] = 0
	image[image < avg] = 0
	
	image[image >= avg] = 1
	
	for _ in range(10):
		image = morphology.binary_erosion(image)

	image = morphology.skeletonize_3d(image)
	result = np.multiply(image, 255.0)
	return result

def convertImages(paths):
	""" Iteratively converts all images from list of given paths
	"""
	srcPath, dstPath = paths
	if not os.path.exists(dstPath):
		os.makedirs(dstPath)

	for name, image in yieldImages(srcPath):
		print("{}".format(name), end="\r")

		out = convertImage(image)

		os.makedirs(os.path.dirname(dstPath + "/" + name), exist_ok=True)
		cv2.imwrite(dstPath + "/" + name, out)

def main():
	srcPaths = ["crossValidationPredictions/pred/pred_wo1/test",
				"crossValidationPredictions/pred/pred_wo2/test",
				"crossValidationPredictions/pred/pred_wo3/test",
				"crossValidationPredictions/pred/pred_wo4/test",
				"crossValidationPredictions/pred/pred_wo5/test",
				"crossValidationPredictions/pred/pred_wo6/test",
				"crossValidationPredictions/pred/pred_wo7/test",
				"crossValidationPredictions/pred/pred_wo8/test",
				"crossValidationPredictions/pred/pred_wo9/test",
				"crossValidationPredictions/pred/pred_wo10/test",
				"crossValidationPredictions/pred/pred_wo11/test",
				"crossValidationPredictions/pred/pred_wo12/test",
				"crossValidationPredictions/pred/pred_wo13/test",
				"crossValidationPredictions/pred/pred_wo14/test",
				"crossValidationPredictions/pred/pred_wo15/test"]

	dstPaths = ["crossValidationPredictions/boundaries_original_algo_3d_skeletonize/nonStrictVersion/boundary_wo1/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo2/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo3/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo4/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo5/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo6/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo7/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo8/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo9/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo10/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo11/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo12/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo13/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo14/test",
				"crossValidationPredictions/boundaries_original_algo_3d_skeletonize/boundary_strict_wo15/test"]

	p = Pool()
	paths = zip(srcPaths, dstPaths)
	p.map(convertImages, paths)

if __name__ == "__main__":
	main()