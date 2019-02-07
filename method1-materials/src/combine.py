import sys
import os
from PIL import Image

dirA = sys.argv[1]
dirB = sys.argv[2]
dst = sys.argv[3]

for filename in os.listdir(dirA):
    if filename.endswith(".png"):        
        imA = Image.open(os.path.join(dirA, filename))
        widthA, heightA = imA.size
        imB = Image.open(os.path.join(dirB, filename))
        widthB, heightB = imB.size
        assert heightA == heightB
        imB = imB.convert(mode="RGB")        
        
        result = Image.new("RGB", (widthA + widthB, heightA))
        result.paste(imA, (0, 0))
        result.paste(imB, (widthA, 0))
        result.save(os.path.join(dst, filename))
        imA.close()
        imB.close()
        
