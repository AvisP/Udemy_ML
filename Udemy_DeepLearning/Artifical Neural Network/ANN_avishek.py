# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:03:00 2018

@author: apaul11
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])  # indicates column no. 1 is categorical
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the data into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Part 2 : Building ANN
import os
#import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Sequential
from keras.layers import Dense

# Initialzing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu',input_dim =11))

# Adding the second hiden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy')

# Fitting the ANN to training set
classifier.fit(X_train,y_train,batch_size = 10, nb_epoch =100)


# Part 3 : Making predications and evaluating the model

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#
#X_hw = {600,
#        'France',
#        'Male',
#        40,
#        3,
#        60000,
#        2,
#        'Yes',
#        'Yes',
#        50000}

X_hw = sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]]))
X_hw_predict = (classifier.predict(X_hw)>0.5)


# Part -4 : Evaluating, Improving and Training the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu',input_dim =11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss='binary_crossentropy')
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch =100)
accuracies = cross_val_score(estimator = classifier, X =X_train, y=y_train, cv = 10, n_jobs = -1)
mean_accuracies = accuracies.mean()
variance_accuracies = accuracies.std()

# Improving ANN
# Droupout and Regularization to reduce overfitting 

# Tuning ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu',input_dim =11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss='binary_crossentropy')
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25,32],
              'nb_epoch': [100, 500],
              'optimizer': ['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv =10)
grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_