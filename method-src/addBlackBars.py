import os
from os import listdir
from os.path import isfile, join
from PIL import Image

""" Script to add black borders around the predicted 1280x1024 image 
	to produce the original image of dimension 1920x1080
"""

path = "realTestSetPredictions/run11/boundaries_gauss"
""" Path to input image folder with subdirs (i.e x1/ x2/ x3/ ...) containing
	images with dimension 1280x1024
"""

new_size = (1920, 1080)
""" Target size
"""

def yieldImages(path):
	""" Yields images from the source folder to convert
		path: Path to the source image folder with subdirectories containg images 
	"""
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