import tensorflow as tf
import numpy as np 
from PIL import Image
from os import listdir
from os.path import isfile, join
from keras.models import *
from keras.initializers import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau
from keras import backend as keras
from modified_keras_preprocessing import image
from modified_keras_preprocessing.image import ImageDataGenerator
import scipy as sp
import imageio
import matplotlib.pyplot as plt
import sys
import cv2
import time
import os
import random
import math
import datetime

#--------------------------------------------------------------------------------------------------
#  !!!!!!!!! Keras needs to be modified locally to work correctly with this dataset !!!!!!!!!!!!  #
# use 'pip show keras' to get the keras install location path                                     #
# open the file keras_preprocessing/image.py                                                      #
# inside the load_img method edit:                                                                #
#   this is the wrong method ---->   line 520: img = img.resize(width_height_tuple, resample)     #
# comment this line and replace with:                                                             #
#   bands = img.split()                                                                           #
#   bands = [b.resize(width_height_tuple, resample) for b in bands]                               #
#   img = pil_image.merge(img.mode, bands)                                                        #
#-------------------------------------------------------------------------------------------------#

# Global parameters
targetSize = (320, 320)
baseTrainingPath = "training_1500_small_dist_field"
baseValidationPath = "training_1500_small_dist_field"

epochs = 50
batchSize = 10 # * 8 # * 8 import for Google TPU  # numImages should be divisable by batch size
numImages = 1500 - 75
validationSplit = 0.1 # numImages * validationSplit should be divisable by batch size

# All subfolders in baseTrainingPath that contain images for x -> input; y -> ground truth
xSubfolders = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]
ySubfolders = ["y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10", "y11", "y12", "y13", "y14", "y15"]

unet_netScale = 1 # Scales the net *down* by factor; should be 1 for original unet from the paper or a multiple of 2
unet_netAlpha = 0.1 # Negative slope of all but last relu activation functions !! 0 seems not to work !!
unet_netErrorFunction = "mean_squared_error" # mean absolute error does not converge !! ; "mean_squared_error"

fusionNet_netScale = 2 
fusionNet_netAlpha = 0.2 
fusionNet_netErrorFunction = "mean_squared_error" 

# Custom accuracy to compute correct pixel coverage between 0 - 1
def acc(y_true, y_pred):
    min1 = tf.minimum(y_true, 1)
    min2 = tf.minimum(y_pred, 1)
    eq = tf.equal(min1, min2)
    nonzero = tf.count_nonzero(eq)
    size = tf.size(y_pred,out_type=tf.int64)
    return tf.divide(nonzero, size)

def fusionNet(input_size):
    global fusionNet_netScale
    global fusionNet_netAlpha
    global fusionNet_netErrorFunction

    netScale = fusionNet_netScale
    netAlpha = fusionNet_netAlpha
    netErrorFunction = fusionNet_netErrorFunction

    input_size = input_size + (4,)
    
    n = netScale

    inputs = Input(input_size)

    conv1 = BatchNormalization()(Conv2D(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(inputs))
    # Resid 1
    conv_resid_1 = BatchNormalization()(Conv2D(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv1))
    conv_resid_1 = BatchNormalization()(Conv2D(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_1))
    conv_resid_1 = BatchNormalization()(Conv2D(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_1))
    conv_resid_1 = BatchNormalization()(Conv2D(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_1))
    # Add Resid
    add_1 = add([conv1, conv_resid_1]) 
    conv_last_1 = BatchNormalization()(Conv2D(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(add_1)) # Long skip 1

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv_last_1)
    conv2 = BatchNormalization()(Conv2D(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(pool1))
    # Resid 2
    conv_resid_2 = BatchNormalization()(Conv2D(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv2))
    conv_resid_2 = BatchNormalization()(Conv2D(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_2))
    conv_resid_2 = BatchNormalization()(Conv2D(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_2))
    conv_resid_2 = BatchNormalization()(Conv2D(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_2))
    # Add Resid
    add_2 = add([conv2, conv_resid_2])
    conv_last_2 = BatchNormalization()(Conv2D(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(add_2)) # Long skip 2

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv_last_2)
    conv3 = BatchNormalization()(Conv2D(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(pool2))
    # Resid 3
    conv_resid_3 = BatchNormalization()(Conv2D(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv3))
    conv_resid_3 = BatchNormalization()(Conv2D(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_3))
    conv_resid_3 = BatchNormalization()(Conv2D(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_3))
    conv_resid_3 = BatchNormalization()(Conv2D(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_3))
    # Add Resid
    add_3 = add([conv3, conv_resid_3])
    conv_last_3 = BatchNormalization()(Conv2D(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(add_3)) # Long skip 3

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv_last_3)
    conv4 = BatchNormalization()(Conv2D(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(pool3))
    # Resid 4
    conv_resid_4 = BatchNormalization()(Conv2D(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv4))
    conv_resid_4 = BatchNormalization()(Conv2D(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_4))
    conv_resid_4 = BatchNormalization()(Conv2D(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_4))
    conv_resid_4 = BatchNormalization()(Conv2D(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_4))
    # Add Resid
    add_4 = add([conv4, conv_resid_4])
    conv_last_4 = BatchNormalization()(Conv2D(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(add_4)) # Long skip 4

    # Bridge
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv_last_4)
    conv5 = BatchNormalization()(Conv2D(int(1024 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(pool4))
    # Resid 5
    conv_resid_5 = BatchNormalization()(Conv2D(int(1024 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv5))
    conv_resid_5 = BatchNormalization()(Conv2D(int(1024 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_5))
    conv_resid_5 = BatchNormalization()(Conv2D(int(1024 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_5))
    conv_resid_5 = BatchNormalization()(Conv2D(int(1024 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_5))
    # Add Resid
    add_5 = add([conv5, conv_resid_5])
    conv_last_5 = BatchNormalization()(Conv2D(int(1024 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(add_5))
    
    # Up 1
    up1 = Conv2DTranspose(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same', strides=(2,2))(conv_last_5)
    long_add_1 = add([up1, conv_last_4])

    conv6 = BatchNormalization()(Conv2D(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(long_add_1))
    # Resid 6
    conv_resid_6 = BatchNormalization()(Conv2D(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv6))
    conv_resid_6 = BatchNormalization()(Conv2D(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_6))
    conv_resid_6 = BatchNormalization()(Conv2D(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_6))
    conv_resid_6 = BatchNormalization()(Conv2D(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_6))
    # Add Resid
    add_6 = add([conv6, conv_resid_6])
    conv_last_6 = BatchNormalization()(Conv2D(int(512 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(add_6))

    # Up 2
    up2 = Conv2DTranspose(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same', strides=(2,2))(conv_last_6)
    long_add_2 = add([up2, conv_last_3])

    conv7 = BatchNormalization()(Conv2D(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(long_add_2))
    # Resid 7
    conv_resid_7 = BatchNormalization()(Conv2D(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv7))
    conv_resid_7 = BatchNormalization()(Conv2D(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_7))
    conv_resid_7 = BatchNormalization()(Conv2D(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_7))
    conv_resid_7 = BatchNormalization()(Conv2D(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_7))
    # Add Resid
    add_7 = add([conv7, conv_resid_7])
    conv_last_7 = BatchNormalization()(Conv2D(int(256 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(add_7))

    # Up 3
    up3 = Conv2DTranspose(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same', strides=(2,2))(conv_last_7)
    long_add_3 = add([up3, conv_last_2])

    conv8 = BatchNormalization()(Conv2D(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(long_add_3))
    # Resid 8
    conv_resid_8 = BatchNormalization()(Conv2D(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv8))
    conv_resid_8 = BatchNormalization()(Conv2D(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_8))
    conv_resid_8 = BatchNormalization()(Conv2D(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_8))
    conv_resid_8 = BatchNormalization()(Conv2D(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_8))
    # Add Resid
    add_8 = add([conv8, conv_resid_8])
    conv_last_8 = BatchNormalization()(Conv2D(int(128 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(add_8))

    # Up 4
    up4 = Conv2DTranspose(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same', strides=(2,2))(conv_last_8)
    long_add_4 = add([up4, conv_last_1])

    conv9 = BatchNormalization()(Conv2D(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(long_add_4))
    # Resid 9
    conv_resid_9 = BatchNormalization()(Conv2D(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv9))
    conv_resid_9 = BatchNormalization()(Conv2D(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_9))
    conv_resid_9 = BatchNormalization()(Conv2D(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_9))
    conv_resid_9 = BatchNormalization()(Conv2D(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(conv_resid_9))
    # Add Resid
    add_9 = add([conv9, conv_resid_9])
    conv_last_9 = BatchNormalization()(Conv2D(int(64 / n), 3, activation = "relu", kernel_initializer="he_normal", padding = 'same')(add_9))

    # Last Part
    out = Conv2D(2, 3, activation = "relu", padding = 'same')(conv_last_9)
    out = Conv2D(1, 3, activation = lambda x: keras.relu(x, alpha=0, max_value=265, threshold=0), padding = 'same')(out)

    model = Model(input = inputs, output = out)

    model.compile(optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False), 
        loss = netErrorFunction, 
        metrics = [accuracy])
    
    print(model.summary())

    return model


def unet(input_size):
    global unet_netScale
    global unet_netAlpha
    global unet_netErrorFunction
    global batchSize

    netScale = unet_netScale
    netAlpha = unet_netAlpha
    netErrorFunction = unet_netErrorFunction

    input_size = input_size + (4,)
    
    n = netScale

    inputs = Input(input_size)
    conv1 = Conv2D(int(64 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(inputs)
    conv1 = Conv2D(int(64 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv1)
    conv1 = Conv2D(int(64 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.05)(pool1)
    
    conv2 = Conv2D(int(128 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(pool1)
    conv2 = Conv2D(int(128 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv2)
    conv2 = Conv2D(int(128 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.1)(pool2)
    
    conv3 = Conv2D(int(256 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(pool2)
    conv3 = Conv2D(int(256 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv3)
    conv3 = Conv2D(int(256 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.1)(pool3)
     
    conv4 = Conv2D(int(512 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(pool3)
    conv4 = Conv2D(int(512 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv4)
    conv4 = Conv2D(int(512 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)

    conv5 = Conv2D(int(1024 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(pool4)
    conv5 = Conv2D(int(1024 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv5)
    # conv5 = Conv2D(int(1024 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv5)

    # up6 = Conv2D(int(512 / n), 2, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(UpSampling2D(size = (2,2))(conv5))
    up6 = Conv2DTranspose(int(512 / n), 2, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same', strides=(2,2))(conv5)
    merge6 = concatenate([conv4,up6], axis = 3)
    merge6 = Dropout(0.1)(merge6)
    conv6 = Conv2D(int(512 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(merge6)
    conv6 = Conv2D(int(512 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv6)
    conv6 = Conv2D(int(512 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv6)
    
    # up7 = Conv2D(int(256 / n), 2, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    up7 = Conv2DTranspose(int(256 / n), 2, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same', strides=(2,2))(conv6)
    merge7 = concatenate([conv3,up7], axis = 3)
    merge7 = Dropout(0.1)(merge7)
    conv7 = Conv2D(int(256 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(merge7)
    conv7 = Conv2D(int(256 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv7)
    conv7 = Conv2D(int(256 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv7)
    
    # up8 = Conv2D(int(128 / n), 2, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    up8 = Conv2DTranspose(int(128 / n), 2, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same', strides=(2,2))(conv7)
    merge8 = concatenate([conv2,up8], axis = 3)
    merge8 = Dropout(0.1)(merge8)
    conv8 = Conv2D(int(128 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(merge8)
    conv8 = Conv2D(int(128 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv8)
    conv8 = Conv2D(int(128 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv8)
    
    # up9 = Conv2D(int(64 / n), 2, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    up9 = Conv2DTranspose(int(64 / n), 2, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same', strides=(2,2))(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(int(64 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(merge9)
    conv9 = Conv2D(int(64 / n), 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv9)
    conv9 = Conv2D(2, 3, activation = lambda x: keras.relu(x, alpha=netAlpha), kernel_initializer="he_normal", padding = 'same')(conv9)
    conv10 = Conv2D(1, 1, activation = lambda x: keras.relu(x, alpha=0, max_value=265, threshold=0))(conv9)

    model = Model(inputs = [inputs], outputs = [conv10])

    model.compile(optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0, amsgrad=False), 
        loss = netErrorFunction, 
        metrics = [acc])
    
    print(model.summary())
   
    return model

    
def loadImage(path, colorMode, targetSize):
  return image.img_to_array(image.load_img(path, color_mode=colorMode, target_size=targetSize))
  
# Load all images that are in one folder 
def loadFolder(path, color_mode, target_size):
    result = []
    files = [f for f in listdir(path) if isfile(join(path, f))]

    # Sort to have same order for x and y images
    files = sorted(files)

    for file in files:
        result.append(loadImage(join(path, file), colorMode=color_mode, targetSize=target_size))

    return result

# Load all images into an x and y array -- Deprecated! --> Use directory flow 
def loadData(basePath):
    global targetSize
    xPath = join(basePath, "x")
    yPath = join(basePath, "y")

    print("Loading...")
    xResult = loadFolder(xPath, "rgba", target_size)
    yResult = loadFolder(xPath, "grayscale", target_size)

    return (xResult, yResult)

# Squish output to 0 - 255 range. Minimum output of the unet is always 0 but maximum is unbound
def clipMax(val):
    maximum = max(1, float(np.amax(val)))
    scale = 255.0 / maximum
    return np.multiply(val, scale)

def applyModel(net, testFileList, originalDataPath, weightsPath):
    global targetSize

    print("Applying model to test set ...")

    model = None
    if net == "unet":
        model = unet(targetSize)
    elif net == "fusionNet":
        model = fusionNet(targetSize)
    else:
        print("Error: Unknown net name: {}".format(net))
        return

    # Load weigths file
    model.load_weights(weightsPath)

    # Load test data
    # x = loadFolder("../../test_1_4/kidney_dataset_1/x", "rgba", targetSize)
    # x += loadFolder("../../test_1_4/kidney_dataset_2/x", "rgba", targetSize)
    # x += loadFolder("../../test_1_4/kidney_dataset_3/x", "rgba", targetSize)
    # x += loadFolder("../../test_1_4/kidney_dataset_4/x", "rgba", targetSize)
    
    # x = loadFolder("training_1500_small_dist_field/x1", "rgba", targetSize)
    
    x = []
    testFilePaths = []
    with open(testFileList, "r") as listFile:
      for line in listFile:
        line = line.lstrip().rstrip()
        testFilePaths.append(line)
        pathToImage = originalDataPath + "/" + line
        x.append(loadImage(pathToImage, "rgba", targetSize))
    
    predInput = np.asarray(x)
	
    # Predict input in batches to not consume to much GPU memory
    batchSize = 10

    counter = 0
    lastBegin = 0
    for _ in range(0, len(x), batchSize):
        upper = min(len(x), lastBegin+batchSize)
        result = model.predict(predInput[lastBegin:upper])
        lastBegin += batchSize

        for r in result:
            r = clipMax(r)
            r = r.astype(np.uint8)
            filename = "pred/{}".format(testFilePaths[counter])
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            imageio.imwrite(filename, r)
            counter += 1
            print("{}/{}".format(counter, len(x), end="\r"))

# Logger for trainig history 
class TrainHistory(Callback):
    def __init__(self, epoch=0):
      Callback.__init__(self)
      if epoch != None:
        self.currentEpoch = epoch
      else:
        self.currentEpoch = 0
        
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc') * 100.0)
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_acc') * 100.0)
        self.currentEpoch += 1

        labels = ["Training", "Validation"]

        fig = plt.figure(1)
        ax1 = plt.subplot(121)
        plt.title('Coverage')
        l1 = ax1.plot(self.accuracies, label="Training", color="black")[0]
        l2 = ax1.plot(self.val_accuracies, label="Validation", color="red")[0]
        plt.xlabel('Epoch')
        plt.ylabel('Correct coverage in %')

        ax1.legend(loc="lower right")

        ax2 = plt.subplot(122)
        plt.title('Loss')
        ax2.plot(self.losses, label="Training", color="black")
        ax2.plot(self.val_losses, label="Validation", color="red")
        plt.xlabel('Epoch')
        plt.ylabel('MSE')

        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig("history.png", format="png")
        
        with open("metrics.csv", "a") as metrics:
          metrics.write("{}, {}, {}, {}, {}\n".format(self.currentEpoch, self.losses[-1], self.accuracies[-1], self.val_losses[-1], self.val_accuracies[-1]))

        plt.clf()
        

def train(net, seed=None, leaveOut=None, checkpoint=None, checkpointEpoch=None):
    global targetSize
    global baseTrainingPath
    global batchSize
    global numImages
    global validationSplit
    global epochs

    print("Training model")
    
    model = None
    if net == "unet":
        model = unet(targetSize)
    elif net == "fusionNet":
        model = fusionNet(targetSize)
    else:
        print("Error: Unknown net name: {}".format(net))
        return
    
    if seed == None:
      seed = int(random.random() * (2 ** 32))    # set to known constant number to reproduce augmentation randomness
    print("Using seed: {}".format(seed))
      
    if checkpoint != None:
      assert (checkpointEpoch != None), "Please supply the starting epoch as checkpointEpoch=..."
      print("Loading weights from checkpoint at last seen epoch: {}".format(checkpointEpoch))
      model.load_weights(checkpoint)
      
    if leaveOut != None and leaveOut > 0 and leaveOut < len(xSubfolders):
      print("Leaving out training set number {} -> {} and {}".format(leaveOut, xSubfolders[leaveOut - 1], ySubfolders[leaveOut - 1]))
      del xSubfolders[leaveOut - 1]
      del ySubfolders[leaveOut - 1]

    # Initialize checkpoints for training
    path = None
    if leaveOut != None:
      begin = "net_wo_{}_t_{}".format(leaveOut, seed)
      path = begin + "_set-epoch_{epoch:02d}-l_{loss:.2f}-a_{acc:.2f}-vl_{val_loss:.2f}-va_{val_acc:.2f}.hdf5"
    else:
      begin = "net_full_t_set_{}".format(seed)
      path = begin + "epoch_{epoch:02d}-l_{loss:.2f}-a_{acc:.2f}-vl_{val_loss:.2f}-va_{val_acc:.2f}.hdf5"

    checkpointLoss = ModelCheckpoint(path, monitor='loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
    checkpointAcc = ModelCheckpoint(path, monitor='acc', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
    checkpointValLoss = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
    checkpointValAcc = ModelCheckpoint(path, monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
    historyCallback = TrainHistory(epoch=checkpointEpoch)
    reduce = ReduceLROnPlateau(patience=3, verbose=1)
    callbacks_list = [checkpointLoss, checkpointAcc, checkpointValLoss, checkpointValAcc, historyCallback, reduce]

    datagen_args = dict(
        rotation_range=15,
        fill_mode="constant",
        cval=0,
        vertical_flip=True,
        validation_split=validationSplit)
    numAugments = 2

    x_datagen = ImageDataGenerator(**datagen_args)
    y_datagen = ImageDataGenerator(**datagen_args)

    # assert (numImages % batchSize == 0), "Trainig data size must be dividable by batch size"

    # Data sources
    x_generator = x_datagen.flow_from_directory(
        baseTrainingPath,
        classes=xSubfolders,
        color_mode="rgba",
        target_size=targetSize,
        # save_to_dir='../testTrain/aug',
        # save_prefix="input",
        # shuffle=False,
        class_mode=None,
        batch_size=batchSize,
        subset="training",
        seed=seed)

    x_validation_generator = x_datagen.flow_from_directory(
        baseValidationPath,
        classes=xSubfolders,
        color_mode="rgba",
        target_size=targetSize,
        # save_to_dir='../testTrain/aug',
        # save_prefix="input",
        # shuffle=False,
        class_mode=None,
        batch_size=batchSize,
        subset="validation",
        seed=seed)

    y_generator = y_datagen.flow_from_directory(
        baseTrainingPath,
        classes=ySubfolders,
        color_mode="grayscale",
        target_size=targetSize,
        # save_to_dir='../testTrain/aug',
        # save_prefix="ground_truth",
        # shuffle=False,
        class_mode=None,
        batch_size=batchSize,
        subset="training",
        seed=seed)

    y_validation_generator = y_datagen.flow_from_directory(
        baseValidationPath,
        classes=ySubfolders,
        color_mode="grayscale",
        target_size=targetSize,
        # save_to_dir='../testTrain/aug',
        # save_prefix="input",
        # shuffle=False,
        class_mode=None,
        batch_size=batchSize,
        subset="validation",
        seed=seed)

    # Combine generators into one which yields image and masks
    train_generator = zip(x_generator, y_generator)
    validation_generator = zip(x_validation_generator, y_validation_generator)

    # Mit den folgende aukommentierten Zeilen kann getestet werden, ob die keras preprocessing manipulation funktioniert hat
    # im = Image.open(baseTrainingPath + "/x1/frame000.png")
    
    # Mit next(x_generator)[0] kann das erste bild des mächsten batches manuell abgefragt werden
    # nach der konvertierung mit .astype(unit8) kann das Bild als Text gespeichert werden und manuell überprüft werden, ob
    # bei einem alpha Wert von 0 die RGB Werte ebenfalls 0 sind.
    # Dies ist manchmal normal, wenn das Bild beispielsweise gedreht wird ist der neue Bildausschnitt am Rand immer [0, 0, 0, 0]

    ## im.load()
    ## bands = im.split()
    ## bands = [b.resize(targetSize, Image.LINEAR) for b in bands]
    ## im = Image.merge('RGBA', bands)

    # np.set_printoptions(threshold=np.nan)
    # with open("test.txt", "w") as f:
    #     f.write(str((next(x_generator)[0]).astype(np.uint8)))
        # f.write(str(list(im.resize(targetSize).getdata())))
        # f.write(str(list(im.getdata())))
    # imageio.imwrite("test.png", next(x_generator)[0].astype(np.uint8))


    model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(numImages * (1.0 - validationSplit) / batchSize),
        validation_data=validation_generator,
        validation_steps=math.ceil(numImages * validationSplit / batchSize),
        epochs=epochs,
        callbacks=callbacks_list)
    
    # Old way of training
    # model.fit(x=np.asarray(x), y=np.asarray(y), epochs=100, validation_split=0.1, verbose=1, batch_size=5, callbacks=callbacks_list)

def printUsage():
    print("Use train as first parameter to train network and predict to apply the trained network to test data.")
    print("The second parameter determins wich net to use. Either use unet or fusionNet")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        printUsage()
    else:
        net = sys.argv[2]
        if (sys.argv[1] == "train"):
            train(net)
        elif (sys.argv[1] == "predict"):
            applyModel(net)
        else:
            printUsage() 
	