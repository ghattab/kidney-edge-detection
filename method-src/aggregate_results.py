import argparse
import pandas
import os

def load_csv(path):
	csv = pandas.read_csv(path)
	csv = csv.set_index("Path")
	return csv

def get_avg_row(csv):
	return csv.loc["avg"]

def get_median_row(csv):
	return csv.loc["median"]

def to_float(dataframe):
	for column in dataframe.columns:
		if column == "Dataset":
			continue
		dataframe[column] = dataframe[column].astype(float)

	return dataframe

def append_avg(dataframe):
	columns = list(dataframe.columns)

	avg_row = dataframe[columns].mean(0)
	avg_row["Dataset"] = "avg"

	dataframe = dataframe.append(avg_row, ignore_index=True)
	return dataframe

def aggregate(base_path, datasets, save_path):
	avg_result = None
	median_result = None

	pandas.set_option('precision', 2)

	for dataset in datasets:
		path = os.path.join(base_path, "evaluation_dataset_{}.csv".format(dataset))
		csv = load_csv(path)

		csv.replace(' None', pandas.np.nan, inplace=True)
		csv.replace('None', pandas.np.nan, inplace=True)

		if avg_result is None:
			columns =  ["Dataset"] + list(csv.columns) #[x.lstrip().rstrip() for x in list(csv.columns)]
			avg_result = pandas.DataFrame(columns=columns)
			median_result = pandas.DataFrame(columns=columns)

		avg = dict(get_avg_row(csv))
		median = dict(get_median_row(csv))

		avg["Dataset"] = str(dataset)
		median["Dataset"] = str(dataset)

		avg_result = avg_result.append(avg, ignore_index=True)
		median_result = median_result.append(median, ignore_index=True)


	avg_result = to_float(avg_result)
	median_result = to_float(median_result)

	avg_result = append_avg(avg_result)
	median_result = append_avg(median_result)

	avg_result.set_index("Dataset").to_csv(os.path.join(save_path, "averages_{}_to_{}.csv".format(min(datasets), max(datasets))))
	median_result.set_index("Dataset").to_csv(os.path.join(save_path, "medians_{}_to_{}.csv".format(min(datasets), max(datasets))))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--base_path", type=str, required=True)
	parser.add_argument("--save_path", type=str, required=True)	

	parser.add_argument("--datasets", nargs="+", type=int, default=[x for x in range(1, 21)], help="Test sets to evaluate. Default 1-20")    

	args = parser.parse_args()
	aggregate(args.base_path, args.datasets, args.save_path)


