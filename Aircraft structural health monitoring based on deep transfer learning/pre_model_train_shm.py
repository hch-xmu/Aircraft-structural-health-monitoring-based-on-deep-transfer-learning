# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:59:06 2021

@author: 22809
"""

import keras
import numpy as np
import operator
import time

from keras.layers import Input 
from keras.layers import Convolution1D 
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers.pooling import GlobalAveragePooling1D
from keras.models import Model

import utils
from utils import create_directory
from utils import read_all_datasets
from utils import transform_labels
from utils import save_logs

from constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from constants import UNIVARIATE_DATASET_NAMES as ALL_DATASET_NAMES
from constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES


def build_model(input_shape, nb_classes, pre_model=None):
	input_layer = keras.layers.Input(input_shape)

	conv1 = Convolution1D(128, 8, padding='same',name='conv_1')(input_layer)
	conv1 = keras.layers.normalization.BatchNormalization(name='batch_normalization_1')(conv1)
	conv1 = keras.layers.Activation(activation='relu',name='activation_1')(conv1)

	conv2 = Convolution1D(filters=256, kernel_size=5, padding='same',name='conv_2')(conv1)
	conv2 = keras.layers.normalization.BatchNormalization(name='batch_normalization_2')(conv2)
	conv2 = Activation('relu',name='activation_2')(conv2)
    
	conv3 = Convolution1D(filters=256, kernel_size=5, padding='same',name='conv_3')(conv2)
	conv3 = keras.layers.normalization.BatchNormalization(name='batch_normalization_3')(conv3)
	conv3 = keras.layers.Activation('relu',name='activation_3')(conv3)

	conv4 = Convolution1D(filters=256, kernel_size=5, padding='same',name='conv_4')(conv3)
	conv4 = keras.layers.normalization.BatchNormalization(name='batch_normalization_4')(conv4)
	conv4 = keras.layers.Activation('relu',name='activation_4')(conv4)    
    
	conv5 = Convolution1D(filters=256, kernel_size=5, padding='same',name='conv_5')(conv4)
	conv5 = keras.layers.normalization.BatchNormalization(name='batch_normalization_5')(conv5)
	conv5 = keras.layers.Activation('relu',name='activation_5')(conv5)

	conv6 = Convolution1D(filters=256, kernel_size=5, padding='same',name='conv_6')(conv5)
	conv6 = keras.layers.normalization.BatchNormalization(name='batch_normalization_6')(conv6)
	conv6 = keras.layers.Activation('relu',name='activation_6')(conv6)        


	conv7 = Convolution1D(256, kernel_size=3,padding='same',name='conv_7')(conv6)
	conv7 = keras.layers.normalization.BatchNormalization(name='batch_normalization_7')(conv7)
	conv7 = keras.layers.Activation('relu',name='activation_7')(conv7)
    
    
	gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv7)

	output_layer = keras.layers.Dense(nb_classes, activation='softmax',name='dense_1')(gap_layer)

	model = keras.models.Model(inputs=input_layer, outputs=output_layer)

	if pre_model is not None:

		for i in range(len(model.layers)-1):
			model.layers[i].set_weights(pre_model.layers[i].get_weights())

	

	return model

def train():
    # read train, val and test sets 
    #提取训练集
    x_train = datasets_dict[dataset_name_tranfer][0]
    y_train = datasets_dict[dataset_name_tranfer][1]

    y_true_val = None 
    y_pred_val = None
    #提取测试集
    x_test = datasets_dict[dataset_name_tranfer][-2]
    y_test = datasets_dict[dataset_name_tranfer][-1]
	

    mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

    nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))

    # make the min to zero of labels
    y_train,y_test = transform_labels(y_train,y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)

    # transform the labels from integers to one hot vectors
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    if len(x_train.shape) == 2: # if univariate 
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))

    start_time = time.time()
    # remove last layer to replace with a new one 
    input_shape = (None,x_train.shape[2])
    model = build_model(input_shape, nb_classes)  

    model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(),
		metrics=['accuracy'])
    model.summary()   
    
    if verbose == True: 
        model.summary()

    # b = model.layers[1].get_weights()
    

    hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
		verbose=verbose, validation_data=(x_test,y_test), callbacks=callbacks)

    # a = model.layers[1].get_weights()

    # compare_weights(a,b)



    y_pred = model.predict(x_test)
    # convert the predicted from binary to integer 
    y_pred = np.argmax(y_pred , axis=1)

    duration = time.time()-start_time

    df_metrics = save_logs(write_output_dir, hist, y_pred, y_true,
						   duration,lr=True, y_true_val=y_true_val,
						   y_pred_val=y_pred_val)

    print('df_metrics')
    print(df_metrics)

    keras.backend.clear_session()
    
    
root_dir = 'E:/被动冲击/迁移学习/data/data06'
results_dir = root_dir+'results/fcn/'

batch_size = 16
nb_epochs = 2000
verbose = False

write_dir_root = root_dir+'/transfer-learning-results/'

for archive_name in ARCHIVE_NAMES:
    # read all datasets
    datasets_dict = read_all_datasets(root_dir, archive_name)
    for dataset_name_tranfer in ALL_DATASET_NAMES:
    # get the directory of the model for this current dataset_name
        write_output_dir = results_dir + archive_name+'_for_git' + '/' + dataset_name_tranfer + '/'
        # set model output path
        file_path = write_output_dir + 'best_model_weight.h5'
        # create directory
        create_directory(write_output_dir)
        # reduce learning rate
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
														  patience=50,min_lr=0.0001)
        # model checkpoint
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
					save_best_only=True)
        callbacks=[reduce_lr,model_checkpoint]

        train()
    

