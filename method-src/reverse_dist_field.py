import cv2
import os
import numpy as np
from skimage import morphology
import re
import argparse
import time

def yieldImages(path):
	fileNames = os.listdir(path)
	fileNames = sorted(fileNames)

	for file in fileNames:
		if file.endswith(".png"):
			distFieldImage = cv2.imread(os.path.join(path, file), 0)

			yield file, distFieldImage

def convertImage(image, threshold):
	image = cv2.resize(image, (1280, 1024))
	avg = np.average(image) * 0.9
	
	image[image < threshold] = 0
	image[image < avg] = 0
	
	image[image >= avg] = 1
	
	for _ in range(10):
		image = morphology.binary_erosion(image)

	image = morphology.skeletonize_3d(image)
	result = np.multiply(image, 255.0)
	# result = result.astype(np.uint8)
	return result

def main(args):
	os.makedirs(args.save_folder, exist_ok=True)

	for subset in args.subsets:
		subset_folder = args.subset_foldername.format(subset)
		os.makedirs(os.path.join(args.save_folder, subset_folder), exist_ok=True)
		path_to_images = os.path.join(args.data_path, subset_folder)

		for name, image in yieldImages(path_to_images):
			print("\r Processing: {}".format(name), end="")

			tick = time.time()
			out = convertImage(image, args.threshold)
			tock = time.time()
			print(" Processing time: {:.2f} seconds".format(tock-tick))

			cv2.imwrite(os.path.join(args.save_folder, subset_folder, name), out)

		print("")
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--threshold", type=int, default=40)

	parser.add_argument("--data_path", type=str, required=True)
	parser.add_argument("--subset_foldername", type=str, default="x{}")

	parser.add_argument("--save_folder", type=str, required=True)	

	parser.add_argument("--subsets", nargs="+", type=int, default=[x for x in range(1, 21)], help="Test sets to evaluate. Default 1-20")    

	args = parser.parse_args()

	main(args)