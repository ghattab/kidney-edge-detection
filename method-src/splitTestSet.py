from os import listdir
from os.path import isfile, join
import random
from shutil import copyfile

originalFolder = "training_1500_small_dist_field"

def prepareTestSet(numPick, seed):
  global baseTrainingPath
  global numImages
  
  random.seed(seed)
  
  assert(len(xSubfolders) == len(ySubfolders)), "Subfolder sizes must match"
  
  for xset, yset in zip(xSubfolders, ySubfolders):
    pathToXImages = originalFolder + "/" + xset
    pathToYImages = originalFolder + "/" + yset
    
    trainFiles = [f for f in listdir(pathToXImages) if isfile(join(pathToXImages, f))]
    numAllFiles = len(trainFiles) 
    
    testFiles = []
    for _ in range(numPick):
      choice = random.choice(trainFiles)
      trainFiles.remove(choice)
      testFiles.append(choice)
    
    print(testFiles)
    print(trainFiles)
    
    with open("{}testImages.txt".format(seed), "a") as out:
      for testFile in testFiles:
        out.write("{}\n".format(xset + "/" + testFile))
        os.remove(pathToXImages + "/" + testFile)
        os.remove(pathToYImages + "/" + testFile)
    
    