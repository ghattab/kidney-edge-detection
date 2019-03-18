import os
from os import listdir
from os.path import isfile, join
from PIL import Image
from multiprocessing import Pool

target_size = (1920, 1080)

def fixTarget(target):
	print("Fixing {} ...".format(target))
	for boundary in range(1, 16):
		for dataset in range(1, 21):
			pathToImages = target + "/boundary_wo" + str(boundary) + "/kidney_dataset_" + str(dataset)
			for file in os.listdir(pathToImages):
				if file.endswith(".png"):
					imageFile = join(pathToImages, file)
					old_im = Image.open(imageFile)
					new_im = Image.new("L", target_size)  # already black
					new_im.paste(old_im, (320, 28))

					new_im.save(imageFile)

def main():
	targets = ["fixedSubmission/Ori-NoStrict",
			   "fixedSubmission/Ori-Strict",
			   "fixedSubmission/Ori3DSk-NoStrict",
			   "fixedSubmission/Ori3DSk-Strict",
			   "fixedSubmission/Ori3DSkSp-NoStrict",
			   "fixedSubmission/Ori3DSkSp-Strict",
			   "fixedSubmission/OriSp-NoStrict",
			   "fixedSubmission/OriSp-Strict"]
	p = Pool()
	p.map(fixTarget, targets)

if __name__ == "__main__":
	main()
		