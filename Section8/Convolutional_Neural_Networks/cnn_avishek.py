# Convolutional Neural network

# Part 1 - Building the CNN

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import time
# Initializing CNN

classifier = Sequential()

# Timer start line
start = time.timer()

# Step 1 : Convolution
classifier.add(Conv2D(32,(3,3),activation="relu",input_shape=(64,64,3)))

# Step 2 : Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 : Adding a second convolutional layer

#classifier.add(Conv2D(32,))

# Step - 4 Flattening
classifier.add(Flatten())

# Step 5 - Full connection
classifier.add(Dense(activation = 'relu',units = 128))
classifier.add(Dense(activation = 'sigmoid',units = 1))

# Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(164, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

# Timer end line
end = time.time()
print("Elapsed time in minutes")
print(0.1*round(end-start)/6)

import os
os.system('say "Your program has finished"')


