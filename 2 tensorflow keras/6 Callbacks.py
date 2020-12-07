# ModelCheckpoint
tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor="val_loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None,
    **kwargs
)

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau 

model_checkpoint = ModelCheckpoint(filepath='model.h5',
                                   monitor='val_accuracy',
                                   mode='auto',
                                   save_best_only=True)

early_stop = EarlyStopping(patience=10)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', 
                                         factor=0.2,
                                         patience=5, 
                                         min_lr=0.000001)
