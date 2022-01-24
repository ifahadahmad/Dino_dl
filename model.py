import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Dropout,Activation,AveragePooling2D,MaxPool2D,GlobalAveragePooling2D

from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model

IMG_SIZE = (80,60)


def dino_model(input_shape=IMG_SIZE):

	input_shape = input_shape + (2,)

	inputs = Input(shape=input_shape)


	X = Conv2D(16,(4,4),strides=2,padding='valid',kernel_initializer='glorot_normal')(inputs)
	X = BatchNormalization()(X)
	X = Activation('elu')(X)
	X = Dropout(0.3)(X)
	X = MaxPool2D(pool_size=(2,2),strides=1,padding='same')(X)


	X = Conv2D(32,(3,3),strides=1,padding='same',kernel_initializer='glorot_normal')(X)
	X = BatchNormalization()(X)
	X = Activation('elu')(X)
	X = Dropout(0.2)(X)
	X = MaxPool2D(pool_size=(3,3),strides=1)(X)


	X = Conv2D(64,(3,3),strides=1,padding='same',kernel_initializer='glorot_normal')(X)
	X = BatchNormalization()(X)
	X = Activation('elu')(X)
	X = Dropout(0.3)(X)
	X = AveragePooling2D(pool_size=(3,3),strides=2)(X)


	X = Conv2D(64,(3,3),strides=2,padding='valid',kernel_initializer='glorot_normal')(X)
	X = BatchNormalization()(X)
	X = Activation('elu')(X)
	X = Dropout(0.4)(X)
	X = AveragePooling2D(pool_size=(2,2),strides=1,padding='same')(X)

	X = Conv2D(128,(1,1),strides=1,padding='same',kernel_initializer='glorot_normal')(X)
	X = BatchNormalization()(X)
	X = Activation('elu')(X)
	X = Dropout(0.3)(X)
	X = GlobalAveragePooling2D()(X)


	X = Dense(512,activation='elu')(X)
	X = Dropout(0.2)(X)

	X = Dense(512,activation='elu')(X)
	X = Dropout(0.5)(X)

	X = Dense(128,activation='elu')(X)
	X = Dropout(0.4)(X)

	output = Dense(3,activation='softmax')(X)

	model = Model(inputs=inputs,outputs=output)
	return model

