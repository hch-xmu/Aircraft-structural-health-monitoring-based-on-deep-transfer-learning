# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:36:41 2021
移植部分层参数，参数可调
@author: 22809
"""

 # this is a transfer learning example 
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

	conv4 = Convolution1D(filters=256, kernel_size=5, padding='same',name='conv_41')(conv3)
	conv4 = keras.layers.normalization.BatchNormalization(name='batch_normalization_41')(conv4)
	conv4 = keras.layers.Activation('relu',name='activation_41')(conv4)    
    
	conv5 = Convolution1D(filters=256, kernel_size=5, padding='same',name='conv_51')(conv4)
	conv5 = keras.layers.normalization.BatchNormalization(name='batch_normalization_51')(conv5)
	conv5 = keras.layers.Activation('relu',name='activation_51')(conv5)

	conv6 = Convolution1D(filters=256, kernel_size=5, padding='same',name='conv_61')(conv5)
	conv6 = keras.layers.normalization.BatchNormalization(name='batch_normalization_61')(conv6)
	conv6 = keras.layers.Activation('relu',name='activation_61')(conv6)        


	conv7 = Convolution1D(256, kernel_size=3,padding='same',name='conv_71')(conv6)
	conv7 = keras.layers.normalization.BatchNormalization(name='batch_normalization_71')(conv7)
	conv7 = keras.layers.Activation('relu',name='activation_71')(conv7)
    
    
	gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv7)

	output_layer = keras.layers.Dense(nb_classes, activation='softmax',name='dense_11')(gap_layer)

	model = keras.models.Model(inputs=input_layer, outputs=output_layer)


	return model

def train(dir_a):
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
	model.load_weights(dir_a+'best_model_weight.h5',by_name=True)
#	weight2=model.get_weights()
	for layers in model.layers[0:24]:
         layers.trainable=True
          
#	pre_model = keras.models.load_model('E:/被动冲击/迁移学习/data/data05/transfer-learning-results/UCR_TS_Archive_2015/01XJ/TG16_0104/'+'best_model.hdf5')    
#	weight3=pre_model.get_weights()
   
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
nb_epochs = 1200
verbose = False

write_dir_root = root_dir+'/transfer-learning-results/'


for archive_name in ARCHIVE_NAMES:
		# read all datasets
		datasets_dict = read_all_datasets(root_dir,archive_name)
		# loop through all datasets
		for dataset_name in ALL_DATASET_NAMES:
			# get the directory of the model for this current dataset_name
			output_dir = results_dir+archive_name +'/'+dataset_name+'/'
			# loop through all the datasets to transfer to the learning
			for dataset_name_tranfer in ALL_DATASET_NAMES:
				# check if its the same dataset
				if dataset_name == dataset_name_tranfer:
					continue
				# set the output directory to write new transfer learning results
				write_output_dir = write_dir_root+archive_name+'/'+dataset_name+\
					'/'+dataset_name_tranfer+'/'
				write_output_dir = create_directory(write_output_dir)
				if write_output_dir is None:
					continue
				print('Tranfering from '+dataset_name+' to '+dataset_name_tranfer)
				# load the model to transfer to other datasets               
                # output file path for the new tranfered re-trained model
				file_path = write_output_dir+'best_model.hdf5'
				# callbacks
				# reduce learning rate
				reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
					min_lr=0.0001)
				# model checkpoint
				model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
					save_best_only=True)
				callbacks=[reduce_lr,model_checkpoint]

				train(output_dir)	
 



