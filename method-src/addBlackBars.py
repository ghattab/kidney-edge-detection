import os
from os import listdir
from os.path import isfile, join
from PIL import Image

path = "realTestSetPredictions/run11/boundaries_gauss"

new_size = (1920, 1080)

def yieldImages(path):
	subdirs = [x[0] for x in os.walk(path)]

	for subdir in subdirs[1:]:	
		fileNames = [f for f in listdir(subdir) if isfile(join(subdir, f))]
		fileNames = sorted(fileNames)

		for file in fileNames:
			old_im = Image.open(join(subdir, file))

			yield (subdir + "/" + file, old_im)


for im_name, old_image in yieldImages(path):
	print(im_name)
	new_im = Image.new("L", new_size)  # already black
	new_im.paste(old_image, (320, 28))

	new_im.save(im_name)