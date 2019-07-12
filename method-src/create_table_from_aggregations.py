import os
import argparse
import pandas

def fix_columns(csv):
	columns = [x.lstrip().rstrip() for x in list(csv.columns)]
	csv.columns = columns
	return csv

def load_loocv_csv_avg(base_path, num, name):
	full_path = os.path.join(base_path, "loocv_{}".format(num), "averages_{}.csv".format(name))
	csv = pandas.read_csv(full_path)
	fix_columns(csv)
	return csv

def load_loocv_csv_median(base_path, num, name):
	full_path = os.path.join(base_path, "loocv_{}".format(num), "medians_{}.csv".format(name))
	csv = pandas.read_csv(full_path)
	fix_columns(csv)
	return csv

def main(args):
	result = pandas.DataFrame(columns=["Loocv", 
									   "Avg. Precision", 
									   "Median Precision", 
									   "Avg. Recall", 
									   "Median Recall", 
									   "Avg. SDE", 
									   "Median SDE", 
									   "Avg. BB-IOU", 
									   "Median BB-IOU"]).set_index("Loocv")
	for subset in args.subsets:
		avg_csv = load_loocv_csv_avg(args.base_path, subset, args.name).set_index("Dataset")
		median_csv = load_loocv_csv_median(args.base_path, subset, args.name).set_index("Dataset")

		averages = avg_csv.loc["avg"]
		medians = median_csv.loc["avg"]

		to_insert = dict()
		to_insert["Loocv"] = str(subset)
		to_insert["Avg. Precision"] = averages["Precision"]
		to_insert["Median Precision"] = medians["Precision"]
		to_insert["Avg. Recall"] = averages["Recall"]
		to_insert["Median Recall"] = medians["Recall"]
		to_insert["Avg. SDE"] = averages["SDE"]
		to_insert["Median SDE"] = medians["SDE"]
		to_insert["Avg. BB-IOU"] = averages["BB-IOU"]
		to_insert["Median BB-IOU"] = medians["BB-IOU"]

		result = result.append(to_insert, ignore_index=True)

	result.set_index("Loocv", inplace=True)
	result.loc["avg"] = result.mean()
	result.to_csv(os.path.join(args.save_path, "result_table.csv"), float_format='%.2f')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--base_path", type=str, required=True)
	parser.add_argument("--save_path", type=str, required=True)	
	parser.add_argument("--name", type=str, required=True)	

	parser.add_argument("--subsets", nargs="+", type=int, default=[x for x in range(1, 17)], help="Test subsets to evaluate. Default 1-16")    

	args = parser.parse_args()

	main(args)





