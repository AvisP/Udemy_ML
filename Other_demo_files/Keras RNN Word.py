# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:57:25 2018
Keras RNN word from Dan does video
@author: apaul11
"""

# IOB Inside Outside beginning

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import Recurrent
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import matplotlib.pyplot as plt
plot.ion()

import pickle
train,test,dicts = pickle.load(open("ATIS.pkl",'rb'),encoding='bytes')