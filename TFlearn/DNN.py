# coding: utf-8
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization,batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression


tflearn.config.init_graph(gpu_memory_fraction=0.4)
IMG_SIZE = (64,64)
TEST_RATIO = 0.1
NUM_PERSON = 1000
NUM_CLASS = 3755
NUM_TEST = int((NUM_CLASS*NUM_PERSON)*TEST_RATIO)
NUM_TRAIN = NUM_CLASS*NUM_PERSON - NUM_TEST
BATCH_SIZE = 128
TRAIN_DIR =  '/data0/jiahao/hcl/train'
TEST_DIR =  '/data0/jiahao/hcl/test'
EPOCH=1
DROPOUT_PROB=0.6

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import image_preloader
train_data,train_labels=image_preloader (TRAIN_DIR, image_shape=IMG_SIZE,
                                         mode='folder',normalize=True,grayscale=True,categorical_labels=True,
                                         files_extension=None, filter_channel=False)
test_data,test_labels=tflearn.data_utils.image_preloader (TEST_DIR, image_shape=IMG_SIZE,
                                                          mode='folder',normalize=True,grayscale=True,categorical_labels=True,
                                                          files_extension=None, filter_channel=False)

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(mean=0)
imgaug = tflearn.ImageAugmentation()
imgaug.add_random_rotation (max_angle=20.0)
imgaug.add_random_crop((64, 64),0)
imgaug.add_random_blur (sigma_max=0.3)

network = input_data(shape=[64, 64], name='input',data_preprocessing=img_prep,data_augmentation=imgaug)
network = tflearn.layers.core.reshape(network,[-1,64,64,1])

conv1_1 = conv_2d(network, 16, 2, strides=1, activation='prelu', name='conv1_1')
pool1_1 = max_pool_2d(conv1_1, 2, strides=1)
#---------------------------------------------------------
conv2_1_1 = conv_2d(pool1_1, 64, [4,2], strides=[1,2,1,1], activation='prelu', name='conv2_1_1')
pool2_1_1 = max_pool_2d(conv2_1_1, 2, strides=1)
inception_3_1_reduce = conv_2d(pool2_1_1,16, filter_size=[1,2], strides=[1,1,2,1],activation='prelu', name ='inception_3_1_reduce' )
inception_3_1 = conv_2d(inception_3_1_reduce, 32, filter_size=5, activation='prelu', name= 'inception_3_1')



conv2_2_2 = conv_2d(pool1_1, 64, [2,4], strides=[1,1,2,1], activation='prelu', name='conv2_1_2')
pool2_2_2 = max_pool_2d(conv2_2_2, 2, strides=1)
inception_3_2_reduce = conv_2d(pool2_2_2,16, filter_size=[2,1], strides=[1,2,1,1],activation='prelu', name ='inception_3_2_reduce' )
inception_3_2 = conv_2d(inception_3_2_reduce, 32, filter_size=5, activation='prelu', name= 'inception_3_2')



conv2_1_3 = conv_2d(pool1_1, 64, 2, strides=1, activation='prelu', name='conv2_1_3')
pool2_1_3 = max_pool_2d(conv2_1_3, 2, strides=1)
pool2_1_3 = local_response_normalization(pool2_1_3)
conv3_3 = conv_2d(pool2_1_3, 32, 2, strides=1, activation='prelu', name='conv3_3')
pool3_3 = max_pool_2d(conv3_3, 3, strides=2)

#---------------------------------------------------------
inception_3_output = merge([inception_3_1, inception_3_2, pool3_3], mode='concat', axis=3)
pool4 = max_pool_2d(inception_3_output, kernel_size=2, strides=1)
pool4 = local_response_normalization(pool4)


conv4_1_1 = conv_2d(pool4, 64, [4,2], strides=[1,2,1,1], activation='prelu', name='conv4_1_1')
pool4_1_1 = max_pool_2d(conv4_1_1, 2, strides=1)
pool4_1_1 = local_response_normalization(pool4_1_1)
inception_4_1_reduce = conv_2d(pool4_1_1,16, filter_size=[1,2], strides=[1,1,2,1],activation='prelu', name ='inception_4_1_reduce' )
inception_4_1 = conv_2d(inception_4_1_reduce, 32, filter_size=5, activation='prelu', name= 'inception_4_1')



conv4_2_2 = conv_2d(pool4, 64, [2,4], strides=[1,1,2,1], activation='prelu', name='conv4_1_2')
pool4_2_2 = max_pool_2d(conv4_2_2, 2, strides=1)
pool4_2_2 = local_response_normalization(pool4_2_2)
inception_4_2_reduce = conv_2d(pool4_2_2,16, filter_size=[2,1], strides=[1,2,1,1],activation='prelu', name ='inception_4_2_reduce' )
inception_4_2 = conv_2d(inception_4_2_reduce, 32, filter_size=5, activation='prelu', name= 'inception_4_2')



conv4_1_3 = conv_2d(pool4, 64, 2, strides=1, activation='prelu', name='conv4_1_3')
pool4_1_3 = max_pool_2d(conv4_1_3, 2, strides=1)
pool4_1_3 = local_response_normalization(pool4_1_3)
conv4_3 = conv_2d(pool4_1_3, 32, 2, strides=1, activation='prelu', name='conv4_3')
pool4_3 = max_pool_2d(conv4_3, 3, strides=2)


#---------------------------------------------------------
inception_5_output = merge([inception_4_1, inception_4_2, pool4_3], mode='concat', axis=3)
pool5 = max_pool_2d(inception_5_output, kernel_size=5, strides=3)

conv5 = conv_2d(pool5, 64, 2, strides=1, activation='prelu', name='conv5')
conv5 = local_response_normalization(conv5)
conv5 = conv_2d(conv5, 64, 2, strides=1, activation='prelu')
conv5 = batch_normalization(conv5)

pool6 = avg_pool_2d(conv5, 2, strides=1)

net = dropout(pool6, DROPOUT_PROB)
loss = fully_connected(net, NUM_CLASS,activation='softmax')
network2 = regression(loss, optimizer='momentum',
                      loss='categorical_crossentropy',
                      learning_rate=0.001)
model = tflearn.DNN(network2,tensorboard_dir='/home/guest/dujinhong/model1/', checkpoint_path='/home/guest/dujinhong/model1/',max_checkpoints=1, tensorboard_verbose=0)

model.load('/home/guest/dujinhong/model1/DNN_trainer2',verbose=True)
model.fit(train_data, train_labels, n_epoch=EPOCH,validation_set=0,
                  shuffle=True,
                   show_metric=True, batch_size=BATCH_SIZE, snapshot_epoch=False,snapshot_step=100000,
                run_id='DNN2')
score = model.evaluate(test_data,test_labels,batch_size=BATCH_SIZE)

print('Batch accuarcy: %0.4f%%' % (score[0] * 100))
model.save('/home/guest/dujinhong/model1/DNN_trainer2')
          
          
          
          
