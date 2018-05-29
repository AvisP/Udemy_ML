# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:10:29 2018
Keras as a simplified interface to Tensorflow
@author: apaul11
"""
import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

img = tf.placeholder(tf.float32, shape=(None,784))

## Keras layers to speed up model definition process

from keras.layers import Dense

x = Dense(128,activation='relu')(img)
x = Dense(128,activation='relu')(x)
preds = Dense(10, activation='softmax')(x)

## placeholder for labels and loss function

labels = tf.placeholder(tf.float32, shape=(None,10))

from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels,preds))

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data',one_hot=True)

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables

init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})
    
from keras.metrics import categorical_accuracy as accuracy

acc_value = accuracy(labels,preds)
with sess.as_default():
    x = (acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels}))

import numpy as np    
print(np.count_nonzero(x)/len(x))