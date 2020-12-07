# Image data preprocessing

# https://keras.io/api/preprocessing/

# Use this code to generate train/validate generator from the same folder

# Set train_data_directory
train_data_directory = ''

# import library
from keras.preprocessing.image import ImageDataGenerator

# set parameters
image_size = (224,224)

batch_size = 128

# binary problem class_mode
class_mode = 'binary' # for multi-class classification problem, use: class_mode = 'category' 

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
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
