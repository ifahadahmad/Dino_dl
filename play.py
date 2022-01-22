
import tensorflow as tf
import cv2
import time
import os
import numpy as np
from grab_screen import grab_screen
from keys import releaseKey,doNothing,goUp,goDown
from getkeys import key_check

TX = 5
WIDTH = 80
HEIGHT = 60
LR = 1e-1
EPOCH = 50
MODEL_NAME = 'dino-{}-{}-{}-epochs.h5'.format(LR,'DINO',EPOCH)

# Load the model

model = tf.keras.models.load_model(MODEL_NAME)

# Initialize activations for lstm

a0 = np.zeros((1,64))
c0 = np.zeros((1,64))
def main():
	for i in list(range(1))[::-1]:
		print(i+1)
		time.sleep(1)
	pause = False
	last_time = time.time()
	while(True):
		single_step = []
		# Grab five images
		for i in range(5):
			screen = grab_screen((250,220,650,520))
			screen = cv2.resize(screen,(80,60))
			single_step.append(screen)
			time.sleep(0.02)
		# Convert to numpy and predict the output
		inputs = np.array(single_step)
		inputs = inputs.reshape((1,inputs.shape[0],inputs.shape[2],inputs.shape[1],inputs.shape[3]))
		predict = model.predict([inputs,a0,c0])
		output = np.argmax(predict)
		
		print(output,predict)
		if output==0:
			doNothing()
		if output==1:
			goUp()
		if output==2:
			goDown()

				
if __name__=='__main__':
	main()


