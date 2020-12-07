# build a VGG16 neural network model

# build a model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

VGG_16 = Sequential()
VGG_16.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

VGG_16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

VGG_16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

VGG_16.add(Flatten())

VGG_16.add(Dense(1024,activation="relu"))
VGG_16.add(Dense(1024,activation="relu"))
VGG_16.add(Dense(1, activation="sigmoid"))

VGG_16.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

VGG_16.summary()

#----------------------------------------------
# change 0/1 to cat and dog
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

VGG_16 = Sequential()
VGG_16.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

VGG_16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

VGG_16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

VGG_16.add(Flatten())

VGG_16.add(Dense(1024,activation="relu"))
VGG_16.add(Dense(1024,activation="relu"))
# 2 neron and softmax activation function here
VGG_16.add(Dense(2, activation="softmax"))

model.compile(loss='categorical_crossentropy',  # categorical_crossentropy here
              optimizer='rmsprop', metrics=['accuracy'])

VGG_16.summary()

