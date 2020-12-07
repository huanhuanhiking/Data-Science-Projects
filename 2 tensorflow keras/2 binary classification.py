# build a VGG16 neural network model
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

VGG_16 = Sequential()
#...
VGG_16.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

VGG_16.summary()

#----------------------------------------------
# change 0/1 to cat and dog
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

VGG_16 = Sequential()
#... 
# 2 neron and softmax activation function here
VGG_16.add(Dense(2, activation="softmax"))

model.compile(loss='categorical_crossentropy',  # categorical_crossentropy here
              optimizer='rmsprop', metrics=['accuracy'])

VGG_16.summary()

