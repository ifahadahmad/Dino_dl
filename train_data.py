import numpy as np
from model import dino_model
import tensorflow as tf
from keras import backend as K
import datetime
import os
TX = 5
VERSION=1
WIDTH = 80
HEIGHT = 60
LR = 1e-1
EPOCH = 100

MODEL_NAME = 'dino-{}-ver.h5'.format(VERSION)

train_data = np.load('training_data_v2.npy',allow_pickle=True)




train_data = train_data[:-4000]
test_data = train_data[-4000:]


# Change dataset to numpy array

train_data_X = train_data[:,0]
train_inter = np.array(train_data_X.tolist())
train_data_X = np.array(train_inter[:,0].tolist(),dtype='float32').reshape((-1,WIDTH,HEIGHT,1))
train_data_X /= 255.
train_speed = np.array(train_inter[:,1],dtype='float32')

mean = np.mean(train_speed)
stddev = np.std(train_speed)

train_speed = (train_speed-mean)/stddev

td = np.empty([train_data_X.shape[0],WIDTH,HEIGHT,1])

for i in range(train_speed.shape[0]):
    td[i] = np.full((WIDTH,HEIGHT,1),train_speed[i],dtype='float32')
td = np.concatenate([train_data_X,td],axis=-1)


train_data_Y = train_data[:,1]
train_data_Y = np.array(train_data_Y.tolist())


test_data_X = test_data[:,0]
test_inter = np.array(test_data_X.tolist())
test_data_X = np.array(test_inter[:,0].tolist(),dtype='float32').reshape((-1,WIDTH,HEIGHT,1)) 
test_data_X /=  255.

test_speed = np.array(test_inter[:,1],dtype='float32')

test_speed = (test_speed-mean)/stddev

tt = np.empty([test_data_X.shape[0],WIDTH,HEIGHT,1])

for i in range(test_speed.shape[0]):
    tt[i] = np.full((WIDTH,HEIGHT,1),test_speed[i],dtype='float32')
tt = np.concatenate([test_data_X,tt],axis=-1)



test_data_Y= test_data[:,1]
test_data_Y = np.array(test_data_Y.tolist())
# number of examples

# if already there use it
# 
if os.path.isfile(MODEL_NAME):
    model = tf.keras.models.load_model(MODEL_NAME)
else:
    # or define new one
    model = dino_model((80,60))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.9,beta_2=0.999,decay=1e-5),
                                                    loss='categorical_crossentropy',metrics=['accuracy'])


# For tensorboard

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + MODEL_NAME
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# CHECKPOINT AT EVERY EPOCH
checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_NAME, monitor='loss', verbose=1, save_best_only=True, mode='min')



history = model.fit(td,train_data_Y,epochs=EPOCH,batch_size=32,validation_data=(tt,test_data_Y),validation_batch_size=32,
                    callbacks=[tensorboard_callback,checkpoint])
# Save at last
model.save(MODEL_NAME)