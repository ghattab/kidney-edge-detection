import sys
import os
from PIL import Image

dirA = sys.argv[1]

dst = sys.argv[2]

for filename in os.listdir(dirA):
    if filename.endswith("_d.png"):  
        imDisp = Image.open(os.path.join(dirA, filename))
        widthD, heightD = imDisp.size
        imDisp = imDisp.convert(mode="RGB") 
        
        frameName = filename[0:-6] + ".png"        
        imA = Image.open(os.path.join(dirA, frameName))
        widthA, heightA = imA.size
        
        assert heightA == heightD   
        
        if heightA == 0:
            print("Found empty image: ", frameName)
            assert false
        
        print (frameName)
        result = Image.new("RGB", (widthA + widthD, heightA))
        result.paste(imA, (0, 0))
        result.paste(imDisp, (widthA, 0))
        result.save(os.path.join(dst, frameName))
        
        imA.close()
        imDisp.close()
        
