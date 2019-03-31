import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import spatial
import re
from datetime import datetime
import os
import statistics
from multiprocessing import Pool
import sys
from scipy import ndimage
from skimage import morphology, measure

class SDFError:

    def __init__(self):
        pass

    def get_error(self, ground_truth_frame, entry_frame):

        # skeleton_ground_truth = (ground_truth_frame == 255) * 1
        # skeleton_entry = (entry_frame == 255) * 1

        contour_ground_truth = ground_truth_frame == 1
        contour_entry = entry_frame == 1 

        #generate sdf from contours
        ground_truth_distance_transform = ndimage.distance_transform_edt(np.logical_not(ground_truth_frame))
        entry_frame_distance_transform = ndimage.distance_transform_edt(np.logical_not(entry_frame))
        #sum the sdf indexed by the contour for each and take average

        if np.sum(entry_frame) > 0:
            precision = np.sum( entry_frame_distance_transform[contour_ground_truth] ) / np.sum(entry_frame)
        else:
            precision = None
        if np.sum(ground_truth_frame) > 0:
            recall = np.sum( ground_truth_distance_transform[contour_entry] ) / np.sum(ground_truth_frame)
        else:
            recall = None

        score = None
        if precision is None and recall is None:
            score = 0
        elif precision is None:
            score = 3.0*recall/2
        elif recall is None:
            score = 3.0*precision/2
        else:
        	score = (precision + recall)/2
        return (precision, recall, score)

def set_bounding_boxes_to_one(image):
	result = np.zeros(image.shape, dtype=np.uint8)
	image_labeled = measure.label(image, background=0)

	values = np.unique(image_labeled)
	for val in values[1:]:	# Skip background 
		pixels_of_label = (image_labeled == val) * 1
		slice_x, slice_y = ndimage.find_objects(pixels_of_label)[0]
		result[slice_x, slice_y] = 1	# Set bounding box area to 1 in result image

	return result

def iou_bounding_box(gt, pred):
	bb_gt = set_bounding_boxes_to_one(gt)
	bb_pred = set_bounding_boxes_to_one(pred)

	return iou_with_offset(bb_gt, bb_pred, offset=0)

def iou_with_offset(gt, pred, offset=0):
	dilated_gt = gt
	if offset > 0:
		# Dilate to add symmetrix border around skeletonized gt
		selem = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
		for _ in range(0, offset):
			dilated_gt = morphology.binary_dilation(dilated_gt, selem=selem)

	true_positive_count = np.sum ( (dilated_gt == 1) & (pred == 1) )
	false_positive_count = np.sum ( (dilated_gt != 1) & (pred == 1) )
	false_negative_count = np.sum ( (gt == 1) & (pred != 1) )

	intersection = true_positive_count
	union = true_positive_count + false_negative_count + false_positive_count

	if intersection == 0 and union == 0:
		return 1
	if union == 0:
		return 0

	return intersection / union

# Old IOU calculation
# def getIOU(gt, pred):
# 	gt = (gt == 255) * 1
# 	pred = (pred == 255) * 1

# 	true_positive_count = np.sum ( (gt == 1) & (pred == 1) )
# 	false_positive_count = np.sum ( (gt != 1) & (pred == 1) )
# 	false_negative_count = np.sum ( (gt == 1) & (pred != 1) )

# 	intersection = true_positive_count
# 	union = true_positive_count + false_negative_count + false_positive_count

# 	if union == 0 and intersection == 0:
# 		return 1
# 	elif union == 0:
# 		return 0

# 	return intersection / union

def loadImage(path):
	image = cv2.imread(path, 0)
	if not (image.shape[1] == 1280 and image.shape[0] == 1024):
		image = image[28:28+1024, 320:320+1280]

	return image

def getImagePair(generatedBoundariesPath, generatedBoundariesIdentifier, boundarySubsets, groundTruthPath, groundTruthSubsetIdentifier):
	for boundarySubset in boundarySubsets:
		subsetPath = generatedBoundariesPath + generatedBoundariesIdentifier.format(boundarySubset)

		files = os.listdir(subsetPath)
		files.sort(key=lambda x: os.path.basename(x))

		for file in files:
			if file.endswith(".png"):
				generatedBoundaryFilePath = subsetPath + "/" + file
				gtBoundaryFilePath = groundTruthPath + groundTruthSubsetIdentifier.format(boundarySubset) + "/" + file

				generated = loadImage(generatedBoundaryFilePath)
				groundTruth = loadImage(gtBoundaryFilePath)
				
				yield ("x" + str(boundarySubset) + "/" + file, generated, groundTruth)

def compairImagePair(groundThruthContourPoints, generatedContourPoints):
	sdfError = SDFError()

	return sdfError.get_error(groundThruthContourPoints, generatedContourPoints)

def compareSets(generatedBoundariesPath, generatedBoundariesIdentifier, boundarySubsets, groundTruthPath, groundTruthSubsetIdentifier, csvName):
	with open(csvName + ".csv", "w") as out:
		out.write("Path, Precision, Recall, SDE, BB-IOU, IOU-20, IOU-15, IOU-10, IOU-5, IOU-0\n")
		scoreList = []
		precisionList = []
		recallList = []
		iou0List  = []
		iou5List  = []
		iou10List = []
		iou15List = []
		iou20List = []
		bbiouList = []
		for (namePath, generatedImage, gtImage) in getImagePair(generatedBoundariesPath, generatedBoundariesIdentifier, boundarySubsets, groundTruthPath, groundTruthSubsetIdentifier):

			generatedImage = (generatedImage == 255) * 1
			gtImage = (gtImage == 255) * 1

			precision, recall, score = compairImagePair(gtImage, generatedImage)
			iou0 = iou_with_offset(gtImage, generatedImage, 0)
			iou5 = iou_with_offset(gtImage, generatedImage, 5)
			iou10 = iou_with_offset(gtImage, generatedImage, 10)
			iou15 = iou_with_offset(gtImage, generatedImage, 15)
			iou20 = iou_with_offset(gtImage, generatedImage, 20)
			bbiou = iou_bounding_box(gtImage, generatedImage)

			if recall is not None:
				recallList.append(recall)
			if precision is not None:
				precisionList.append(precision)

			scoreList.append(score)
			iou0List.append(iou0)
			iou5List.append(iou5)
			iou10List.append(iou10)
			iou15List.append(iou15)
			iou20List.append(iou20)
			bbiouList.append(bbiou)

			out.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(namePath, precision, recall, score, bbiou, iou20, iou15, iou10, iou5, iou0))
		avgScore = statistics.mean(scoreList)
		medianScore = statistics.median(scoreList)

		avgPrecision = statistics.mean(precisionList)
		medianPrecision = statistics.median(precisionList)

		avgRecall = statistics.mean(recallList)
		medianRecall = statistics.median(recallList)

		avgIOU20 = statistics.mean(iou20List)
		medianIOU20 = statistics.median(iou20List)
		avgIOU15 = statistics.mean(iou15List)
		medianIOU15 = statistics.median(iou15List)
		avgIOU10 = statistics.mean(iou10List)
		medianIOU10 = statistics.median(iou10List)
		avgIOU5 = statistics.mean(iou5List)
		medianIOU5 = statistics.median(iou5List)
		avgIOU0 = statistics.mean(iou0List)
		medianIOU0 = statistics.median(iou0List)

		avgbbIOU = statistics.mean(bbiouList)
		medianbbIOU = statistics.median(bbiouList)

		out.write("avg, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(avgPrecision, avgRecall, avgScore, avgbbIOU, avgIOU20 ,avgIOU15, avgIOU10, avgIOU5, avgIOU0))
		out.write("median, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(medianPrecision, medianRecall, medianScore, medianbbIOU, medianIOU20, medianIOU15, medianIOU10, medianIOU5, medianIOU0))

def evaluateSubset(params):
	valSubset, algorithm, useStrict, baseDataPath, basegtPath = params

	baseBoundariesPath = None
	if useStrict:
		baseBoundariesPath = baseDataPath + "crossValidationPredictions/{}/strictVersion/".format(algorithm)
	else:
		baseBoundariesPath = baseDataPath + "crossValidationPredictions/{}/nonStrictVersion/".format(algorithm)
	gtPath = basegtPath + "skeletonized_processed_labels/" # "processed_labels/"

	print("Evaluating validation set {}".format(valSubset))
	valSubsetPath = None
	if useStrict:
		valSubsetPath = baseBoundariesPath + "boundary_strict_wo{}/test/".format(str(valSubset))
	else:
		valSubsetPath = baseBoundariesPath + "boundary_wo{}/test/".format(str(valSubset))

	allSubsets = [x for x in range(1, 21)]

	if valSubset in allSubsets: # The test set that was left out during training
		allSubsets.remove(valSubset)

	savePath = None
	if useStrict:
		savePath = baseDataPath + "crossValidationPredictions/evaluations/{}/strictVersion/".format(algorithm)
	else:
		savePath = baseDataPath + "crossValidationPredictions/evaluations/{}/nonStrictVersion/".format(algorithm)

	os.makedirs(savePath, exist_ok=True)

	# Optimize: Cache results from AllExcept_ and use these values for 16-20
	compareSets(valSubsetPath, 
				"x{}", 
				allSubsets, 
				gtPath, 
				"kidney_dataset_{}/croppedGround_truth", 
				savePath + "evalBoundaryWo{}_AllExcept{}".format(valSubset, valSubset))
	compareSets(valSubsetPath, 
				"x{}", 
				[16, 17 ,18, 19, 20], 
				gtPath, 
				"kidney_dataset_{}/croppedGround_truth", 
				savePath + "evalBoundaryWo{}_16_to_20".format(valSubset))
	if valSubset < 16:
		compareSets(valSubsetPath, 
					"x{}", 
					[valSubset], 
					gtPath, 
					"kidney_dataset_{}/croppedGround_truth", 
					savePath + "evalBoundaryWo{}_Only{}".format(valSubset, valSubset))


def main(algorithm, strict, baseDataPath, basegtPath):
	p = Pool()

	valSubsets = [x for x in range(1, 16)]
	algorithmList = [algorithm] * 15
	strictList = [strict] * 15
	baseDataPathList = [baseDataPath]  * 15
	basegtPathList = [basegtPath] * 15

	params = zip(valSubsets, algorithmList, strictList, baseDataPathList, basegtPathList)
	p.map(evaluateSubset, params)

if __name__ == "__main__":

	algorithm = sys.argv[1]
	strict = (sys.argv[2] == "strict")
	baseDataPath = sys.argv[3]
	basegtPath = sys.argv[4]

	main(algorithm, strict, baseDataPath, basegtPath)