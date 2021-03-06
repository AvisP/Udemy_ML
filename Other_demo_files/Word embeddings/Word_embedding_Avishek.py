# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:14:17 2018

@author: apaul11
"""
# Importing data
import os
import sys
import numpy as np

# User defined parameters
TEXT_DATA_DIR = r'C:\Users\apaul11\Documents\Sync\Deep Learning\Word embeddings\20_newsgroup'
GLOVE_DIR = r'C:\Users\apaul11\Documents\Sync\Deep Learning\Word embeddings'
MAX_NB_WORDS = 3
MAX_SEQUENCE_LENGTH = 5
VALIDATION_SPLIT =  0.85
EMBEDDING_DIM = 100

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids


for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)

print('Found %s texts.' % len(texts))


# Format text samples and labels into tensors 

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


# #Preparing the embedding layer

# Compute an index mapping words to known embeddings, by parsing the data dump of pre-trained embeddings

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# Using embedding_index dictionary and our word_index to compute our embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
# load this embedding matrix into an Embedding layer. Note that we set trainable=False to prevent the weights from being updated during training
        
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

## Training a 1D covnet

from keras.models import Model
from keras.layers import Input,Dense,Flatten,MaxPooling1D,Conv1D
from keras.layers import GlobalMaxPooling1D


#sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
sequence_input = Input(shape=(256,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(pool_size=5,strides=None)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(pool_size=5,strides=None)(x)
x = Conv1D(128, 5, activation='relu')(x)
#x = MaxPooling1D(pool_size=35,strides=None)(x)  # global max pooling#
x = GlobalMaxPooling1D()(x) # global maxpooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=2, batch_size=128)
