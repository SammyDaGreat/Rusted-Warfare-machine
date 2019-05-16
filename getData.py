import numpy as np
from PIL import ImageGrab
import cv2
import time,math,os,json
import win32gui, win32ui, win32con, win32api    # win32 libs

file_iter = 0
WIDTH = 960
HEIGHT = 940

### Adjust UUID before running ###

uuid = -1

# Thoughts:
#	Should I record mouse pos all the time or just when clicking? 
#	How to distinguish clicking and dragging?
#	Higher fps?

### Adjust UUID before running ###

def is_window(name):
	title = win32gui.GetWindowText(win32gui.GetForegroundWindow())
	if title == name: return True

def is_rusted():
	return is_window('Rusted Warfare')

def process_img(original_image):
	#processed = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
	processed = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
	#processed = cv2.Canny(processed,threshold1 = 170,threshold2=300)
	return processed

def save_y(n,img,mouse,kb):
	file_name = str(uuid)+'.'+str(n)
	mouse_array = np.array(mouse)
	kb_array = np.array(kb)
	np.save('Data\\x\\'+file_name,img)
	np.save('Data\\y1\\'+file_name,mouse_array)
	np.save('Data\\y2\\'+file_name,kb_array)

def replay(that_uuid):
	try:
		i=0
		while(True):
				time.sleep(0.1)
				shot = np.load('Data//x//'+str(that_uuid)+'.'+str(i)+'.npy')
				cv2.imshow('window',shot)
				mouse_pos = np.load('Data//y1//'+str(that_uuid)+'.'+str(i)+'.npy')
				kb = np.load('Data//y2//'+str(that_uuid)+'.'+str(i)+'.npy')
				i += 1
				if cv2.waitKey(25)&0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
	except:
		return

f = open('VK_CODE','r')
VK_CODE = json.load(f)
f.close()
vk = [VK_CODE[key] for key in VK_CODE]


def get_kb():
	kb = [0,0]
	if win32api.GetKeyState(1)<0:
		kb[0] = 1
	if win32api.GetKeyState(2)<0:
		kb[1] = 1
	return kb
'''
# Returns one-hot list for keyboard status. Pressed => 1, Unpressed => 0
def get_kb():
	kb = [0 for i in vk]
	for i in range(len(vk)):
		if win32api.GetKeyState(vk[i])<0:
			kb[i] = 1
	return kb
	'''

# Returns standardized mouse pos
## Should I actually standardize here?
def get_mouse():
	cords = win32gui.GetCursorPos()
	mouse_pos = [cords[0]/WIDTH,cords[1]/HEIGHT]
	return mouse_pos

uuid_init = 0
if uuid==-1:
	while os.path.exists('Data//x//'+str(uuid_init)+'.0.npy'):
		uuid_init += 1
	uuid = uuid_init

# Main body
#replay(0)
last_time = time.time()
while(True):
	while(win32api.GetKeyState(112)>=0) or not is_rusted():
		time.sleep(0.1)
	while(win32api.GetKeyState(113)>=0) and is_rusted():
		time.sleep(0.015)
		screen = np.array(ImageGrab.grab(bbox=(0,40,WIDTH,40+HEIGHT)))
		new_screen = process_img(screen)
		last_time = time.time()

		mouse_pos = get_mouse()
		kb = get_kb()

		if uuid >= 0:
			save_y(file_iter,new_screen,mouse_pos,kb)
			file_iter = file_iter + 1
		print(file_iter)
		cv2.imshow('window',new_screen)
		if cv2.waitKey(25)&0xFF == ord('q'):
			cv2.destroyAllWindows()
			break