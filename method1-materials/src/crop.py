import sys
import os
from PIL import Image

dir = sys.argv[1]

for filename in os.listdir(dir):
    if filename.endswith(".png"):        
        im = Image.open(os.path.join(dir, filename))
        width, height = im.size
        if width > 1264:
            im = im.crop((328, 37, 1592, 1047))
            im.save(os.path.join(dir, filename))
        im.close()
        
