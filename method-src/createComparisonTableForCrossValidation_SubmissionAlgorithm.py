from pandas import DataFrame
import pandas 
import os 

targets = [("boundaries_original_algo_3d_skeletonize/nonStrictVersion", "Ori 3D Skel Non Strict"),
		   ("boundaries_original_algo_3d_skeletonize_spur_removal/nonStrictVersion", "Ori 3D Skel Spur Removal Non Strict"),
		   
		   ("boundaries_original_algo_3d_skeletonize_spur_removal/strictVersion", "Ori 3D Skel Spur Removal Strict"),
		   ("boundaries_original_algo_3d_skeletonize/strictVersion", "Ori 3D Skel Strict"),

		   ("boundaries_original_algo/nonStrictVersion", "Ori Non Strict"),
		   ("boundaries_original_algo_spur_removal/nonStrictVersion", "Ori Spur Removal Non Strict"),

		   ("boundaries_original_algo_spur_removal/strictVersion", "Ori Spur Removal Strict"),
		   ("boundaries_original_algo/strictVersion", "Ori Strict"),
		   ]

basePath = "crossValidationPredictions/evaluations_with_iou_steps_and_bb/"

with open("crossEvalComparisons_step_IOU_and_BB_IOU.csv", "w") as outFile:
	outFile.write("Method and Subset, Avg. Precision, Median Precision, Avg. Recall, Median Recall, Avg. SDE, Median SDE, Avg. BB-IOU, Median BB-IOU, Avg. IOU-20, Median IOU-20, Avg. IOU-15, Median IOU-15, Avg. IOU-10, Median IOU-10, Avg. IOU-5, Median IOU-5, Avg. IOU-0, Median IOU-0\n")
	for target in targets:
		targetPath, targetName = target
		pathToCSVs = basePath + targetPath
		files = os.listdir(pathToCSVs)
		files.sort()
		subsets = [x for x in range(1, 16)]
		for currentSet in subsets:
			file = "evalBoundaryWo{}_16_to_20.csv".format(currentSet)
			csvData = pandas.read_csv(pathToCSVs + "/" + file, names=["Path", "Precision", "Recall", "Score", "BB-IOU", "IOU-20", "IOU-15", "IOU-10", "IOU-5", "IOU-0"])
			
			avgRow = csvData.loc[(csvData["Path"] == "avg")]
			avgScore = float(avgRow["Score"].values[0])
			avgPrecision = float(avgRow["Precision"].values[0])
			avgRecall = float(avgRow["Recall"].values[0])
			avgBBIOU = float(avgRow["BB-IOU"].values[0])
			avgIOU_20 = float(avgRow["IOU-20"].values[0])
			avgIOU_15 = float(avgRow["IOU-15"].values[0])
			avgIOU_10 = float(avgRow["IOU-10"].values[0])
			avgIOU_5 = float(avgRow["IOU-5"].values[0])
			avgIOU_0 = float(avgRow["IOU-0"].values[0])

			medianRow = csvData.loc[(csvData["Path"] == "median")]
			medianScore = float(medianRow["Score"].values[0])
			medianPrecision = float(medianRow["Precision"].values[0])
			medianRecall = float(medianRow["Recall"].values[0])
			medianBBIOU = float(medianRow["BB-IOU"].values[0])
			medianIOU_20 = float(medianRow["IOU-20"].values[0])
			medianIOU_15 = float(medianRow["IOU-15"].values[0])
			medianIOU_10 = float(medianRow["IOU-10"].values[0])
			medianIOU_5 = float(medianRow["IOU-5"].values[0])
			medianIOU_0 = float(medianRow["IOU-0"].values[0])

			methodAndSubset = targetName + " " + str(currentSet)
			outLine = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(methodAndSubset, avgPrecision, medianPrecision, avgRecall, medianRecall, avgScore, medianScore, avgBBIOU, medianBBIOU, avgIOU_20, medianIOU_20, avgIOU_15, medianIOU_15, avgIOU_10, medianIOU_10, avgIOU_5, medianIOU_5, avgIOU_0, medianIOU_0)
			outFile.write(outLine)

			currentSet += 1
		outFile.write("avg, , , , , , , , , , , , , , , , , , \n")