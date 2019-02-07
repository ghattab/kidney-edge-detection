import sys
import os
from PIL import Image

dir = sys.argv[1]

ctr = 0
for filename in os.listdir(dir):
    if filename.endswith(".png"):
        print(filename)        
        im = Image.open(os.path.join(dir, filename))
        width, height = im.size
        assert width == (3 * 1280) and height == 1024, "width: %r, height: %r" % (width, height)
        im = im.convert(mode="RGB")    
        im.verify()
        im.close()
        ctr = ctr + 1
print("images checked: ", ctr)

        
