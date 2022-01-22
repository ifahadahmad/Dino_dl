import cv2
import time
import os
import numpy as np
from grab_screen import grab_screen
from getkeys import key_check


# Script for collecting data

def keys_to_output(keys):

	output = [0,0,0]

	if 0x26 in keys:
		output[1] = 1
	elif 0x28 in keys:
		output[2] = 1
	else:
		output[0] = 1
	return output


file_name = 'training_data_2.npy'


# If data exist load it otherwise create new one
if os.path.isfile(file_name):
	print('File exists,loading previous data!')
	training_data = list(np.load(file_name,allow_pickle=True))
else:
	print("File does not exist creating fresh")
	training_data = []



def main():

	for i in list(range(4))[::-1]:
		print(i+1)
		time.sleep(1)
	pause = False
	while(True):
		if not pause:
			single_step = []
			keys = key_check()
			output = keys_to_output(keys)
			# Collect image for each time step
			for i in range(5):
				screen = grab_screen((250,220,650,520))
				screen = cv2.resize(screen,(80,60))		
				keys = key_check()
				if 0x28 in keys or 0x26 in keys:
					output = keys_to_output(keys)	
				single_step.append(screen)
				time.sleep(0.02)

			training_data.append([single_step,output])

			
			if len(training_data)%200 == 0:
				print(len(training_data))
				np.save(file_name,training_data)
		key = key_check()
		if 0x54 in key:
			if pause:
				pause = False
				time.sleep(0.4)
			else:
				pause = True
				print("paused")
				time.sleep(0.4)


if __name__=='__main__':
	main()