import tensorflow as tf
import numpy as np 
from PIL import Image
from os import listdir
from os.path import isfile, join

from tensorflow.keras.models import *
from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau
from tensorflow.keras import backend as keras

from modified_keras_preprocessing import image
from modified_keras_preprocessing.image import ImageDataGenerator
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import random
import math
import datetime
import argparse
import re

#--------------------------------------------------------------------------------------------------
#  !!!!!!!!! Keras needs to be modified locally to work correctly with this dataset !!!!!!!!!!!!  #
# use ´pip show keras´ to get the keras install location path                                     #
# open the file keras_preprocessing/image.py                                                      #
# inside the load_img method edit:                                                                #
#   this is the wrong method ---->   line 520: img = img.resize(width_height_tuple, resample)     #
# comment this line and replace with:                                                             #
#   bands = img.split()                                                                           #
#   bands = [b.resize(width_height_tuple, resample) for b in bands]                               #
#   img = pil_image.merge(img.mode, bands)                                                        #
#-------------------------------------------------------------------------------------------------#

# Custom accuracy to compute correct pixel coverage between 0 - 1
def acc(y_true, y_pred):
    min1 = tf.minimum(y_true, 1)
    min2 = tf.minimum(y_pred, 1)
    eq = tf.equal(min1, min2)
    nonzero = tf.count_nonzero(eq)
    size = tf.size(y_pred,out_type=tf.int64)
    return tf.divide(nonzero, size)

def schedule(epoch, lr):
    if epoch == 0:
        return 0.001
    else:
        return 0.0001

def vanilla_unet(input_size):
    input_size = input_size + (4,)

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = lambda x: keras.relu(x, alpha=0, max_value=255, threshold=0))(conv9)

    model = Model(inputs = [inputs], outputs = [conv10])

    model.compile(optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0, amsgrad=False), 
                                   loss = "mse", 
                                   metrics = [acc])

    print(model.summary())

    return model

def unet(input_size, netScale, netAlpha, batch_size, netErrorFunction="mse"):
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

    first_number = re.findall(r'\d+', files[0])[0]
    start_offset = int(first_number)

    for file in files:
        result.append(loadImage(join(path, file), colorMode=color_mode, targetSize=target_size))

    return result, start_offset

# Load all images into an x and y array -- Deprecated! --> Use directory flow 
def loadData(basePath):
    global targetSize
    xPath = join(basePath, "x")
    yPath = join(basePath, "y")

    print("Loading...")
    xResult, _ = loadFolder(xPath, "rgba", target_size)
    yResult, _ = loadFolder(xPath, "grayscale", target_size)

    return (xResult, yResult)

# Squish output to 0 - 255 range. Minimum output of the unet is always 0 but maximum is unbound
def clipMax(val):
    maximum = max(1, float(np.amax(val)))
    scale = 255.0 / maximum
    return np.multiply(val, scale)

def applyModel(model, args):
    assert(args.test_save)
    assert(args.data_path_test)

    targetSize = (320, 320)

    print("Applying model to test set(s) ...")

    batchSize = args.batch_size

    if args.weights_file:
        print("Using weights from weights file")
        loocv_weight_paths = []
        with open(args.weights_file, "r") as weights_file:
            for line in weights_file: # Weights must be consecutive from 1 to n
                if line.startswith("#"): # Ignore possible comments
                    continue
                weights_part = line.split(":")[1]
                if weights_part == "SKIP":   # Loocv can be skipped if no weights are avialable
                    loocv_weight_paths.append(None)
                else:    
                    loocv_weight_paths.append(weights_part.lstrip().rstrip())

        print("Loaded weight paths")
        for loocv_num, weight_path in enumerate(loocv_weight_paths):
            if weights_part == None:
                print("Skipping {}".format(loocv_num + 1))
                continue

            print("Predicting loocv {}".format(loocv_num + 1))
            loocv_folder_name = "loocv_{}".format(loocv_num + 1)
            print("Loading weights")
            model.load_weights(weight_path)

            print("Predicting subsets...")
            for subset in args.test_sets:
                subset_folder_name = "x{}".format(subset)
                x, start_offset = loadFolder(os.path.join(args.data_path_test, subset_folder_name), "rgba", targetSize)
                
                predInput = np.asarray(x)

                counter = start_offset
                lastBegin = 0
                for _ in range(0, len(x), batchSize):
                    upper = min(len(x), lastBegin+batchSize)
                    tick = time.time()
                    result = model.predict(predInput[lastBegin:upper])
                    tock = time.time()
                    print("Inference took: {:.5f} seconds".format(tock-tick))
                    lastBegin += batchSize

                    for r in result:
                        r = clipMax(r)
                        r = r.astype(np.uint8)
                        frame_name = "frame{:03d}.png".format(counter)
                        filename = os.path.join(args.test_save, loocv_folder_name, subset_folder_name, frame_name)
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        imageio.imwrite(filename, r)
                        counter += 1
                        print("\rx{}: {}/{}".format(subset, counter, start_offset + len(x)), end="")

                print("")

    else:
        assert(args.weights)
        print("Using specified weights")
        model.load_weights(args.weights)
        
        for subset in args.test_sets:
            subset_folder_name = "x{}".format(subset)
            x, start_offset = loadFolder(os.path.join(args.data_path_test, subset_folder_name), "rgba", targetSize)
            
            predInput = np.asarray(x)

            counter = start_offset
            lastBegin = 0
            for _ in range(0, len(x), batchSize):
                upper = min(len(x), lastBegin+batchSize)
                result = model.predict(predInput[lastBegin:upper])
                lastBegin += batchSize

                for r in result:
                    r = clipMax(r)
                    r = r.astype(np.uint8)
                    frame_name = "frame{:03d}.png".format(counter)
                    filename = os.path.join(args.test_save, subset_folder_name, frame_name)
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    imageio.imwrite(filename, r)
                    counter += 1
                    print("{}/{}".format(counter, start_offset + len(x), end="\r"))

# Logger for trainig history 
class TrainHistory(Callback):
    def __init__(self, save_dir, epoch=None):
      Callback.__init__(self)
      if epoch != None:
        self.currentEpoch = epoch
      else:
        self.currentEpoch = 0
      self.save_dir = save_dir
        
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
        plt.savefig(os.path.join(self.save_dir, "history.png"), format="png")
        
        with open(os.path.join(self.save_dir, "metrics.csv"), "a") as metrics:
          metrics.write("{}, {}, {}, {}, {}\n".format(self.currentEpoch, self.losses[-1], self.accuracies[-1], self.val_losses[-1], self.val_accuracies[-1]))

        plt.clf()
    

def main(args):
    targetSize = (320, 320)

    if args.seed == None:
        if args.seed_file:
            seeds = []
            with open(args.seed_file, "r") as file_with_seeds:    
                for line in file_with_seeds:
                    seeds.append(int(line.split(":")[1].lstrip().rstrip()))
            args.seed = seeds[args.leave_out - 1]
        else:
            args.seed = int(random.random() * (2 ** 32))    # set to known constant number to reproduce augmentation randomness
    print("Using seed: {}".format(args.seed))

    # Initialize all random generators
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Use single thread for reproducability
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.set_session(sess)

    model = unet(targetSize, netScale=args.net_scale, netAlpha=args.net_alpha, batch_size=args.batch_size)
    # print("!!USING VANILLA UNET!!")
    # model = vanilla_unet(targetSize)

    if args.mode == "test":
        applyModel(model, args)
        return

    assert (args.validation_split < 1)

    xSubfolders = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"] 
    ySubfolders = ["y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10", "y11", "y12", "y13", "y14", "y15"] 

    print("Training model")

    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, "seed.txt"), "w") as seed_file:
        seed_file.write("{}".format(args.seed))
        
    if args.leave_out and args.leave_out <= 15:
        print("Leaving out subset {}".format(args.leave_out))
        xSubfolders.remove("x{}".format(args.leave_out))
        ySubfolders.remove("y{}".format(args.leave_out))
        assert (len(xSubfolders) == len(ySubfolders) == 14)

        print("Using x folders: {}".format(xSubfolders))
        print("Using y folders: {}".format(ySubfolders))

    if args.num_images == None:
        args.num_images = 100 * len(xSubfolders)

    weights_dir = os.path.join(args.save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    path = os.path.join(weights_dir, "epoch_{epoch:02d}-l_{loss:.2f}-a_{acc:.2f}-vl_{val_loss:.2f}-va_{val_acc:.2f}.hdf5")
    
    checkpointLoss = ModelCheckpoint(path, monitor='loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
    checkpointAcc = ModelCheckpoint(path, monitor='acc', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
    checkpointValLoss = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
    checkpointValAcc = ModelCheckpoint(path, monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
    historyCallback = TrainHistory(save_dir=args.save_dir)
    
    # learning_rate_scheduler = LearningRateScheduler(schedule=schedule, verbose=1)
    lr_reduce = ReduceLROnPlateau(patience=3, verbose=1)

    callbacks_list = [checkpointLoss, checkpointAcc, checkpointValLoss, checkpointValAcc, historyCallback, lr_reduce]

    datagen_args = dict(
        rotation_range=15,
        fill_mode="constant",
        cval=0,
        vertical_flip=True,
        validation_split=args.validation_split)

    x_datagen = ImageDataGenerator(**datagen_args)
    y_datagen = ImageDataGenerator(**datagen_args)

    # Data sources
    x_generator = x_datagen.flow_from_directory(
        args.data_path_input,
        classes=xSubfolders,
        color_mode="rgba",
        target_size=targetSize,
        class_mode=None,
        batch_size=args.batch_size,
        subset="training",
        seed=args.seed)

    x_validation_generator = x_datagen.flow_from_directory(
        args.data_path_input,
        classes=xSubfolders,
        color_mode="rgba",
        target_size=targetSize,
        class_mode=None,
        batch_size=args.batch_size,
        subset="validation",
        seed=args.seed)

    y_generator = y_datagen.flow_from_directory(
        args.data_path_target,
        classes=ySubfolders,
        color_mode="grayscale",
        target_size=targetSize,
        class_mode=None,
        batch_size=args.batch_size,
        subset="training",
        seed=args.seed)

    y_validation_generator = y_datagen.flow_from_directory(
        args.data_path_target,
        classes=ySubfolders,
        color_mode="grayscale",
        target_size=targetSize,
        class_mode=None,
        batch_size=args.batch_size,
        subset="validation",
        seed=args.seed)

    # Combine generators into one which yields image and masks
    train_generator = zip(x_generator, y_generator)
    validation_generator = zip(x_validation_generator, y_validation_generator)

    def yield_from_generator(generator):
        while True:
            x, y = next(generator)
            yield (x, y)

    model.fit_generator(
        yield_from_generator(train_generator),
        steps_per_epoch=math.ceil(args.num_images * (1.0 - args.validation_split) / args.batch_size),
        validation_data=yield_from_generator(validation_generator),
        validation_steps=math.ceil(args.num_images * args.validation_split / args.batch_size),
        epochs=args.epochs,
        callbacks=callbacks_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--mode", type=str, default="train", help="Mode of the script. 'train' for training and 'test' for testing. Default 'train'")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights used for inference in 'train' mode")
    parser.add_argument("--test_sets", nargs="+", type=int, default=[x for x in range(1, 21)], help="Test sets to predict. Default 1-20")
    parser.add_argument("--test_save", type=str, default=None, help="Path to save predicted images after testing")
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="Batch size. Default 10")
    parser.add_argument("-di", "--data_path_input", type=str, help="Path to training directory containing x1, ..., x15 (inputs)")
    parser.add_argument("-dt", "--data_path_target", type=str, help="Path to target label directory containing y1, ..., y15")
    parser.add_argument("-dtest", "--data_path_test", type=str, help="Path to test directory containing x1, ..., x20")
    parser.add_argument("-vs", "--validation_split", type=float, default=0.1, help="Validation split. Default 0.1")
    parser.add_argument("--net_scale", type=float, default=1, help="Downscales net by specified factor. Default 1")
    parser.add_argument("--net_alpha", type=float, default=0.1, help="Alpha values for LeakyReLU activations. Default 0.1")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("-n", "--num_images", type=int, default=1500, help="Number of training images")
    parser.add_argument("-loocv", "--leave_out", type=int, default=None, help="Which training set to leave out")
    parser.add_argument("--save_dir", type=str, help="Save folder for weights and metrics")
    parser.add_argument("--seed", type=int, default=None, help="Random initialization seed")
    parser.add_argument("--seed_file", type=str, default=None, help="Path to file with seeds for random number generation.")
    parser.add_argument("--weights_file", type=str, default=None, help="Path to file with weights for each loocv.")

    args = parser.parse_args()

    main(args)
	