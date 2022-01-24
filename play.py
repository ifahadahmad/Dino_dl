import tensorflow as tf
import cv2
import time
import os
import numpy as np
from grab_screen import grab_screen
from keys import releaseKey,doNothing,goUp,goDown
from getkeys import key_check
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("file:///C:/Users/ahmad/Desktop/dino.html")

VERSION = 1
WIDTH = 80
HEIGHT = 60
LR = 1e-1
EPOCH = 50
MODEL_NAME = 'dino-{}-ver.h5'.format(VERSION)

# Load the model

model = tf.keras.models.load_model(MODEL_NAME)

# Initialize activations for lstm
train_data = np.load('training_data_v1.npy',allow_pickle=True)




train_data = train_data[:-4000]



train_data_X = train_data[:,0]
train_inter = np.array(train_data_X.tolist())
train_speed = np.array(train_inter[:,1],dtype='float32')
mean = np.mean(train_speed)
stddev = np.std(train_speed)


def main():
	for i in list(range(1))[::-1]:
		print(i+1)
		time.sleep(1)
	pause = False
	# last_time = time.time()
	while(True):
		if not pause:
			screen = grab_screen((215,125,635,365))
			screen = cv2.resize(screen,(80,60))
			# Convert to numpy and predict the output
			inputs = np.array(screen,dtype='float32')
			inputs /= 255.
			inputs = inputs.reshape((1,inputs.shape[1],inputs.shape[0],1))
			speed = driver.execute_script("return Runner.instance_.currentSpeed")
			# speed = np.array(speed,dtype='float64').reshape(1,)
			speed = (speed-mean)/stddev
			speed = np.full((1,WIDTH,HEIGHT,1),speed,dtype='float32')
			inputs = np.concatenate([inputs,speed],axis=-1)
			predict = model.predict(inputs)
			output = np.argmax(predict)
			print(output,predict)
			if output==0:
				doNothing()
			if output==1:
				goUp()
			if output==2:
				goDown()
		keys = key_check()
		if 0x54 in keys:
			if pause:
				pause = False
				time.sleep(0.5)
			else:
				pause = True
				doNothing()
				time.sleep(0.5)

	driver.close()
if __name__=='__main__':
	main()