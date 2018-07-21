# coding: utf-8
from __future__ import division, print_function, absolute_import

import keras
from keras.layers import GaussianNoise, Input, Dense, Dropout, Flatten, Concatenate, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, flow_from_directory

IMG_SIZE = (64,64)
TEST_RATIO = 0.1
NUM_PERSON = 1000
NUM_CLASS = 3755
NUM_TEST = int((NUM_CLASS*NUM_PERSON)*TEST_RATIO)
NUM_TRAIN = NUM_CLASS*NUM_PERSON - NUM_TEST
BATCH_SIZE = 64
TRAIN_DIR = r'data/train'
TEST_DIR = r'data/test'
EPOCH = 1
DROPOUT_PROB = 1.0
from PIL import Image
import numpy as np

# 类别-汉字索引
import pickle
with open('demo1/word.pkl', 'rb') as word:
    word_index = pickle.load(word)


# Data Generators
train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        validation_split=0.1)
test_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
)
train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        color_mode = 'grayscale',
        target_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = 'categorical',
        shuffle = True, seed = 0, subset = 'training', interpolation = 'bilinear')
validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        color_mode = 'grayscale',
        target_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = 'categorical',
        shuffle = True, seed = 0, subset = 'validation', interpolation = 'bilinear')
test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        color_mode = 'grayscale',
        target_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = 'categorical',
        shuffle = True, seed = 0, interpolation = 'bilinear')

# Network Structure
input_0 = Input(input_shape = (64,64,1), name = 'input_0')
input_1 = GaussianNoise(stddev = 0.3, name = 'input_1')(input_0)


conv1_1 = Conv2D(filters = 16, kernel_size = 2, strides = 1,
                 activation = 'prelu', padding='valid', name='conv1_1')(input_1)
pool1_1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(conv1_1)

## The 1st Channel
conv2_1_1 = Conv2D(filters = 64, kernel_size = (4, 2), strides = (2, 1),
                 activation = 'prelu', padding='valid', name='conv2_1_1')(pool1_1)
pool2_1_1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(conv2_1_1)
inception_3_1_reduce= Conv2D(filters = 16, kernel_size = (1, 2), strides = (1, 2),
                             activation = 'prelu', padding='valid', name='inception_3_1_reduce')(pool2_1_1)
inception_3_1 = Conv2D(filters = 32, kernel_size = 5, strides = 1,
                        activation = 'prelu', padding='valid', name='inception_3_1_reduce')(inception_3_1_reduce)


## The 2nd Channel
conv2_2_2 = Conv2D(filters = 64, kernel_size = (2, 4), strides = (1, 2),
                 activation = 'prelu', padding='valid', name='conv2_2_2')(pool1_1)
pool2_2_2 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(conv2_2_2)
inception_3_2_reduce= Conv2D(filters = 16, kernel_size = (2, 1), strides = (2, 1),
                             activation = 'prelu', padding='valid', name='inception_3_2_reduce')(pool2_2_2)
inception_3_2 = Conv2D(filters = 32, kernel_size = 5, strides = 1,
                        activation = 'prelu', padding='valid', name='inception_3_2')(inception_3_2_reduce)

## The 3rd Channel
conv2_1_3 = Conv2D(filters = 64, kernel_size = 2, strides = 1,
                 activation = 'prelu', padding='valid', name='conv2_1_3')(pool1_1)
pool2_1_3 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(conv2_1_3)
pool2_1_3 = BatchNormalization(pool2_1_3)
conv3_3 = Conv2D(filters = 32, kernel_size = 2, strides = 1,
                 activation = 'prelu', padding='valid', name='conv3_3')(pool2_1_3)
pool3_3 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(conv3_3)

## Merge
inception_3_output = Concatenate()([inception_3_1, inception_3_2, pool3_3])
pool4 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(inception_3_output)
pool4 = BatchNormalization(pool4)

## The 1st Channel
conv4_1_1 = Conv2D(filters = 64, kernel_size = (4, 2), strides = (2, 1),
                 activation = 'prelu', padding='valid', name='conv4_1_1')(pool4)
pool4_1_1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(conv4_1_1)
pool4_1_1 = BatchNormalization(pool4_1_1)
inception_4_1_reduce = Conv2D(filters = 16, kernel_size = (1, 2), strides = (1, 2),
                             activation = 'prelu', padding='valid', name='inception_4_1_reduce')(pool4_1_1)
inception_4_1 = Conv2D(filters = 32, kernel_size = 5, strides = 1,
                        activation = 'prelu', padding='valid', name='inception_4_1')(inception_4_1_reduce)

## The 2nd Channel
conv4_2_2 = Conv2D(filters = 64, kernel_size = (2, 4), strides = (1, 2),
                 activation = 'prelu', padding='valid', name='conv4_2_2')(pool4)
pool4_2_2 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(conv4_2_2)
pool4_2_2 = BatchNormalization(pool4_2_2)
inception_4_2_reduce= Conv2D(filters = 16, kernel_size = (2, 1), strides = (2, 1),
                             activation = 'prelu', padding='valid', name='inception_4_2_reduce')(pool4_2_2)
inception_4_2 = Conv2D(filters = 32, kernel_size = 5, strides = 1,
                        activation = 'prelu', padding='valid', name='inception_4_2')(inception_4_2_reduce)

## The 3rd Channel
conv4_1_3 = Conv2D(filters = 64, kernel_size = 2, strides = 1,
                 activation = 'prelu', padding='valid', name='conv4_1_3')(pool4)
pool4_1_3 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(conv4_1_3)
pool4_1_3 = BatchNormalization(pool4_1_3)
conv4_3 = Conv2D(filters = 32, kernel_size = 2, strides = 1,
                 activation = 'prelu', padding='valid', name='conv4_3')(pool4_1_3)
pool4_3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv4_3)

## Merge
inception_5_output = Concatenate()([inception_4_1, inception_4_2, pool4_3])
pool5 = MaxPooling2D(pool_size = (5, 5), strides = 3, padding = 'valid')(inception_5_output)


conv5 = Conv2D(filters = 64, kernel_size = 2, strides = 1,
                 activation = 'prelu', padding='valid', name='conv5')(pool5)
conv5 = BatchNormalization(conv5)
conv5 =  Conv2D(filters = 64, kernel_size = 2, strides = 1,
                 activation = 'prelu', padding='valid')(conv5)
conv5 = BatchNormalization(conv5)
pool6 = AveragePooling2D(pool_size = (2, 2), strides = 1, padding = 'valid')(inception_5_output)

net = Dropout(DROPOUT_PROB)(pool6)
output = Dense(NUM_CLASS, activation='softmax')(net)


# Building Model
model = Model(inputs = input_0, outputs = output)
sgd = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = False)
model.compile(optimizer = sgd,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Add Checkpoint
## TO DO
callbacks = []


# Training
model.fit_generator(
        train_generator,
        epochs = EPOCH,
        callbacks = callbacks,
        validation_data = validation_generator,
        validation_steps = 800)
