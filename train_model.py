import numpy as np
from dinoModel import dino_model
import tensorflow as tf
from keras import backend as K
import datetime
import os
TX = 5
WIDTH = 80
HEIGHT = 60
LR = 1e-1
EPOCH = 50
MODEL_NAME = 'dino-{}-{}-{}-epochs.h5'.format(LR,'DINO',EPOCH)

train_data = np.load('training_data_v5.npy',allow_pickle=True)




train_data = train_data[:-300]
test_data = train_data[-300:]


# Change dataset to numpy array

train_data_X = train_data[:,0]
train_data_X = np.array(train_data_X[:].tolist()).reshape((-1,TX,WIDTH,HEIGHT,3))
train_data_Y = train_data[:,1]
train_data_Y = np.array(train_data_Y[:].tolist())


test_data_X = test_data[:,0]
test_data_X = np.array(test_data_X[:].tolist()).reshape((-1,TX,WIDTH,HEIGHT,3)) 
test_data_Y= test_data[:,1]
test_data_Y = np.array(test_data_Y[:].tolist())


# number of examples

m = train_data_X.shape[0]
a0 = np.zeros((m,64))
c0 = np.zeros((m,64))

a1 = np.zeros((300,64))
c1 = np.zeros((300,64))

# if already there use it

if os.path.isfile(MODEL_NAME):
    model = tf.keras.models.load_model(MODEL_NAME)
else:
    # or define new one
    model = dino_model(5,(80,60))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1,beta_1=0.9,beta_2=0.999,decay=0.01),
                                                    loss='categorical_crossentropy',metrics=['accuracy'])


# For tensorboard

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + MODEL_NAME
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# CHECKPOINT AT EVERY EPOCH
checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_NAME, monitor='loss', verbose=1, save_best_only=True, mode='min')



history = model.fit([train_data_X,a0,c0],train_data_Y,epochs=EPOCH,batch_size=32,validation_data=([test_data_X,a1,c1],test_data_Y),validation_batch_size=16,
                    callbacks=[tensorboard_callback,checkpoint])
# Save at last
model.save(MODEL_NAME)

