import tensorflow as tf
import keras
import numpy as np
from PIL import ImageGrab
import cv2
import time,math
import win32gui, win32ui, win32con, win32api    # win32 libs
import matplotlib.pyplot as plt
import os
from keras import utils as np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image

model_name = 'Cycle 0.1'
img_size = [576,778]

def is_window(name):
    title = win32gui.GetWindowText(win32gui.GetForegroundWindow())
    if title == name: return True

def is_slither():
    return is_window('slither.io - Google Chrome')

def fast():
    win32api.keybd_event(32,0,0,0) #32 is spacebar
def slow():
    win32api.keybd_event(32,0,win32con.KEYEVENTF_KEYUP,0)  

def process_img(original_image):
    processed = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    return processed

def alpha2cords(alpha):
	r = 50
	a = alpha*2*math.pi
	if alpha<=0.25:
		dx = math.cos(a)
		dy = math.sin(a)
	elif 0.25<alpha<=0.5:
		a = math.pi-a
		dx = -math.cos(a)
		dy = math.sin(a)
	elif 0.5<alpha<=0.75:
		a = a-math.pi
		dx = -math.cos(a)
		dy = -math.sin(a)
	elif 0.75<alpha:
		a = math.pi*2-a
		dx = math.cos(a)
		dy = -math.sin(a)
	x = 288+dx*r
	y = 476-dy*r	#y axis on monitor is inversed
	return[x,y]

'''
#Test if the numpy array is correctly read
img = train_data[0][20].reshape(img_size[1],img_size[0])
plt.imshow(img)
plt.show()
'''

model = keras.models.load_model(model_name)
optimizer = tf.train.RMSPropOptimizer(0.001)
model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])

#main body
last_time = time.time()
while(True):
	print('ready')
	time.sleep(0.2)
	while win32api.GetKeyState(0x43)<0 and is_slither():
		screen = np.array(ImageGrab.grab(bbox=(0,85,576,863))) #576*778=448128
		new_screen = process_img(screen) # canny

		x_input = new_screen.reshape(1,img_size[1],img_size[0],1) # resize to fit conv net
		x_input = x_input.astype('float32')
		x_input /= 225

		y1 = model.predict(x_input)[0]
		print('>\t'+str(int(y1*360)))

        #print('Loop took {} seconds'.format(time.time()-last_time))
        #last_time = time.time()

		[x,y] = alpha2cords(y1)
		win32api.SetCursorPos([int(x),int(y)])

		time.sleep(0.01)
