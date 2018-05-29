# -*- coding: utf-8 -*-
"""
Created on Wed May  2 20:48:25 2018
Keras on minist
@author: apaul11
"""

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import matplotlib.pyplot as plt
plt.ion()

(X_train, y_train),(X_test,y_test) = mnist.load_data()

X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0],'train_samples')
print(X_test.shape[0],'test_samples')

# Convert class vectors to binary class matrices
nb_classes = 10
Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes)

model = Sequential()
model.add(Dense(128, input_shape=(784,)))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

# to check  model status while adding layers use model.summary()

sgd = SGD()
model.compile(loss='categorical_crossentropy',optimizer=sgd)
h = model.fit(X_train,Y_train,batch_size=128, nb_epoch=3, show_accuracy=True,validation_data=(X_test,y_test),verbose=1)

# to see metrics type h.history
# to get the weights type model.get_weights()

# Plotting weights
W1,b1,W2,b2 = model.get_weights()

sx, sy = (16,8)
f,con  = plt.subplots(sx,sy, sharex='col',sharey='row')
for xx in range(sx):
    for yy in range(sy):
        con[xx,yy].pcolormesh(W1[:,8*xx+yy].reshape(28,28))


sx, sy = (5,2)
f,con  = plt.subplots(sx,sy, sharex='col',sharey='row')
for xx in range(sx):
    for yy in range(sy):
        con[xx,yy].pcolormesh(W1[:,5*xx+yy].reshape(28,28))
        
## Model visualization using keras
        # Install pydot_ng
from keras.utils import plot_model
plot_model(model, to_file='model.png')
# dipsly using $dipslay model.png

