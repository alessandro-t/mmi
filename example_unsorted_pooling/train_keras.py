# coding: utf-8

from keras.models import Sequential
from keras.layers import Dropout, BatchNormalization, Activation, Dense, MaxPool2D, Conv2D, Flatten

import h5py
import keras
import numpy as np

w = 105

xxx = h5py.File('data/data_w_%d.h5' % w)
X_train = xxx['train/X'][:]
y_train = xxx['train/y'][:]
X_test = xxx['test/X'][:]
y_test = xxx['test/y'][:]


model = Sequential()

layers = [
    Conv2D(32, (7, 7), activation='linear', input_shape=(w, w, 1)),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.5),
    Conv2D(64, (5, 5), activation='linear'),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.5),
    Conv2D(64, (5, 5), activation='linear'),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(1024),
    Activation('relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
]

for layer in layers:
    model.add(layer)

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X_train, y_train, batch_size=20, epochs=20)

#model.save('models/train_%d' % w)
#model.save_weights('models/weights_%d' % w)

res = model.predict(X_test)
acc = 100.*(res.argmax(axis=1) == y_test.argmax(axis=1)).sum()/len(res)

print('%.5f' % acc)
