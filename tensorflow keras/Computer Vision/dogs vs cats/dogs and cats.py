# Using plaidml.keras.backend backend
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras


# Set train_data_directory
train_data_directory = 'E:/cats and dogs/train'

# import library
from keras.preprocessing.image import ImageDataGenerator

# set parameters
image_size = (112,112)

batch_size = 128

# binary problem class_mode
class_mode = 'binary' # for multi-class classification problem, use: class_mode = 'category'

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.33 # set validation split
    )

train_generator = datagen.flow_from_directory(
    train_data_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=class_mode, # for multi-class classification problem, use 'category'
    subset='training') # set as training data

validation_generator = datagen.flow_from_directory(
    train_data_directory, # same directory as training data
    target_size=image_size,
    batch_size=batch_size,
    class_mode=class_mode, # for multi-class classification problem, use 'category'
    subset='validation') # set as validation data


# building a VGG16 net
# build a model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization

VGG_16 = Sequential()
VGG_16.add(Conv2D(input_shape=(112,112,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
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

VGG_16.add(Flatten())

VGG_16.add(Dense(1024,activation="relu"))
VGG_16.add(Dense(1, activation="sigmoid"))

VGG_16.summary()

# compile model
from keras import optimizers

VGG_16.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Callbacks

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint("model.h5",
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

early_stopping = EarlyStopping(monitor='val_acc',
                               min_delta=0,
                               patience=5,
                               verbose=1,
                               mode='auto')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=1,
                                            verbose=1,
                                            factor=0.1,
                                            min_lr=0.00001)
nb_epochs = 1000

history = VGG_16.fit_generator(train_generator,
                               steps_per_epoch = train_generator.samples // batch_size,
                               validation_data = validation_generator,
                               validation_steps = validation_generator.samples // batch_size,
                               epochs = nb_epochs,
                               callbacks=[checkpoint, early_stopping, learning_rate_reduction],
                               verbose=2)