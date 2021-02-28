
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 21:58:40 2021

@author: sbasak
"""

"""
Drone signals for single-signal classification scenario
The drone signal can be downloaded from this link
https://kuleuven-my.sharepoint.com/:u:/g/personal/sanjoy_basak_kuleuven_be/EerXZTSHi5ZCj2i_PED6X4sBIGkQqlFQiyRpGAVzQIjBOQ?e=6q23OJ
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import h5py


from tensorflow.python.client import device_lib
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Add, Dense, Activation
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
print(device_lib.list_local_devices())


classes=["dx4e","dx6i","MTx","Nineeg","Parrot","q205","S500","tello","WiFi","wltoys"]
mods = classes
maxlen=256


filename1 = 'GenDataRSNT_10_snrss2_NW.h5' # load the dataset 
h5f = h5py.File(filename1, 'r')

X_train_t = h5f['X_train']
Y_train_t = h5f['labels_RSNT_F_train']

train_idx_t = h5f['train_idx']
X_test_t = h5f['X_test']
Y_test_t = h5f['labels_RSNT_F_test']
test_idx_t = h5f['test_idx']

X_train=np.array(X_train_t[()])
X_test=np.array(X_test_t[()])
Y_train=np.array(Y_train_t[()])
Y_test=np.array(Y_test_t[()])
train_idx=np.array(train_idx_t[()])
test_idx=np.array(test_idx_t[()])
h5f.close()

print("--"*10)
print("Training data size:",X_train.shape)
print("Training labels size:",Y_train.shape)
print("Testing data size:",X_test.shape)
print("Testing labels size:",Y_test.shape)
print("--"*10)


def residual_stack(x, f):
    x = Conv2D(f, 1, strides=1, padding='same')(x)
    x = Activation('linear')(x)
    
    # residual unit 1    
    x_shortcut = x
    x = Conv2D(f, 3, strides=1, padding="same")(x)
    x = Activation('relu')(x)
    x = Conv2D(f, 3, strides=1, padding="same")(x)
    
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    # residual unit 2    
    x_shortcut = x
    x = Conv2D(f, 3, strides=1, padding="same")(x)
    x = Activation('relu')(x)
    x = Conv2D(f, 3, strides = 1, padding = "same")(x)


    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x


# define resnet model
def ResNet(input_shape, classes):   
    x_input = Input(input_shape)
    x = x_input

    num_filters = 32
    x = residual_stack(x, num_filters) #1
    x = MaxPooling2D(pool_size=2, strides=None, padding='valid')(x)
    x = residual_stack(x, num_filters) #2
    x = MaxPooling2D(pool_size=2, strides=None, padding='valid')(x)
    x = residual_stack(x, num_filters) #3
    x = MaxPooling2D(pool_size=2, strides=None, padding='valid')(x)
    x = residual_stack(x, num_filters) #4
    x = MaxPooling2D(pool_size=2, strides=None, padding='valid')(x)
    x = residual_stack(x, num_filters) #5
    x = MaxPooling2D(pool_size=2, strides=None, padding='valid')(x)
    x = residual_stack(x, num_filters) #6
    

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    x = Flatten()(x)
    x = Dense(classes , activation='softmax', kernel_initializer = glorot_uniform(seed=0))(x)
    

    model = Model(inputs = x_input, outputs = x)
    return model


# checkpoint
filepath="spectrogramweights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# initialize and train model
train=True

if train:
    model = ResNet((maxlen, maxlen,1), len(mods))
    adm = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, Y_train, epochs = 200, batch_size = 32,
                    callbacks=callbacks_list, validation_data=(X_test, Y_test))
else:
    model = load_model(filepath)
    model.summary()
    history = model.fit(X_train, Y_train, epochs = 100, batch_size = 32,
                    callbacks=callbacks_list, validation_data=(X_test, Y_test))



# evaluate model on test data
loss, acc = model.evaluate(X_test, Y_test, batch_size=32)
print('EVALUATING MODEL ON TEST DATA:')
print('Test Accuracy: ', str(round(acc*100, 2)), '%')
print('\n')






