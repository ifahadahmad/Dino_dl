import cv2
import time
import os
import numpy as np
from grab_screen import grab_screen
from getkeys import key_check
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("file:///C:/Users/ahmad/Desktop/dino.html")

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


file_name = 'training_data2.npy'


# If data exist load it otherwise create new one
if os.path.isfile(file_name):
	print('File exists,loading previous data!')
	training_data = list(np.load(file_name,allow_pickle=True))
else:
	print("File does not exist creating fresh")
	training_data = []



def main():

	for i in list(range(6))[::-1]:
		print(i+1)
		time.sleep(1)
	pause = False
	# last = time.time()
	while(True):
		screen = grab_screen((215,125,635,365))
		screen = cv2.resize(screen,(80,60))
		keys = key_check()
		output = keys_to_output(keys)
		speed = driver.execute_script("return Runner.instance_.currentSpeed")
		training_data.append([[screen,speed],output])
		# print("time is ",time.time()-last)
		# last = time.time()
		if len(training_data)%500==0:
			print(len(training_data))
			np.save(file_name,training_data)
		time.sleep(0.02)
	driver.close()

if __name__=='__main__':
	main()