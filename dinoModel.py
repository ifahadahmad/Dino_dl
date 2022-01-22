import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model

n_a = 64
preprocess_input =  tf.keras.applications.vgg19.preprocess_input
reshaper = tfl.Reshape((1,-1))
IMG_SIZE = (80,60)
LSTM_cell = tfl.LSTM(n_a,return_state=True,name="Lstm1stlayer")


def dino_model(Tx,input_size=IMG_SIZE):

# define input shape -> shape is (None,WIDTH,HEIGHT,3)
	
	input_shape = input_size + (3,)

# using pretrained vgg19

	base_model =  tf.keras.applications.vgg19.VGG19(input_shape=input_shape,
													include_top=False,
													weights='imagenet')
	# Freeze weights
	base_model.trainable = False


	# Input layer
	X = tf.keras.Input(shape=((Tx,)+input_shape))


	# LSTM initial activations
	a0 = tf.keras.Input(shape=(n_a,),name='a0')
	c0 = tf.keras.Input(shape=(n_a,),name='c0')
	a = a0
	c = c0


	for t in range(Tx):
		# Preprocess image for vgg19
		x = preprocess_input(X[:,t])
		# Calculate activations of vgg19
		x = base_model(x,training=False)
		# Pooling followed by Dropout
		x = tfl.GlobalAveragePooling2D(name="pooling"+str(t+1))(x)
		x = tfl.Dropout(0.6,name="dropout"+str(t+1))(x)
		# Reshape for LSTM
		x = reshaper(x)
		# Single time-step for LSTM
		a,_,c = LSTM_cell(inputs=x,initial_state=[a,c])
	# Softmax layer
	output = tfl.Dense(3,activation="softmax",name="Softmax")(a)
	model = Model(inputs=[X,a0,c0],outputs=output)
	return model

