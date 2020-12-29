# Using plaidml.keras.backend backend
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

print(keras.__version__)

# set parameters
image_size = (224,224)

batch_size = 64

# binary problem class_mode
class_mode = 'categorical'
# for binary classification problem, use: class_mode = 'binary'
# for multi-class classification problem, use: class_mode = 'categorical'


# build the model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Conv2D, MaxPool2D

# import vgg16 model
from keras.applications.vgg16 import VGG16

vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

vgg16_conv.trainable = False

model = Sequential()

model.add(vgg16_conv)

model.add(Flatten())
model.add(Dense(1024,activation="relu"))
model.add(Dense(1024,activation="relu"))


model.add(Dense(47, activation="softmax"))

print(model.summary())


# Set train_data_directory
train_data_directory = 'D:/data science/Dog Breeds'

# import library
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=0.5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2 # set validation split
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



# compile model
from keras import optimizers

model.compile(loss='categorical_crossentropy',
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
                                            min_lr=0.00000001)


nb_epochs = 1000

# from keras.models import load_model
 
# load model
# model = load_model("C:/Users/jiahu/Documents/model.h5")

history = model.fit_generator(train_generator,
                               steps_per_epoch = train_generator.samples // batch_size,
                               validation_data = validation_generator,
                               validation_steps = validation_generator.samples // batch_size,
                               epochs = nb_epochs,
                               callbacks=[checkpoint, early_stopping, learning_rate_reduction])


# Unfreeze the base model
model.trainable = True

# retrain all parameter within the model
history = model.fit_generator(train_generator,
                               steps_per_epoch = train_generator.samples // batch_size,
                               validation_data = validation_generator,
                               validation_steps = validation_generator.samples // batch_size,
                               epochs = nb_epochs,
                               callbacks=[checkpoint, early_stopping, learning_rate_reduction])