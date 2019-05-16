import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from keras import utils as np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image

path_x = 'Data\\x\\'
path_y1 = 'Data\\y1\\'
path_y2 = 'Data\\y2\\'
img_size = [576,778]

imlist = os.listdir(path_x)[:500]
num_samples = len(imlist)

'''
#Test if the numpy array is correctly read
im1 = np.load(path_x + imlist[0])
plt.imshow(im1)
plt.show()
'''

immatrix = np.array([np.load(path_x + file_name).flatten() for file_name in imlist],'f')
label = np.array([np.load(path_y1 + file_name).flatten() for file_name in imlist],'f')

data,Label = shuffle(immatrix,label,random_state = 2)
train_data = [data,Label]
print('x_train shape, y_train shape:\t',train_data[0].shape,train_data[1].shape)

'''
#Test if the numpy array is correctly read
img = train_data[0][20].reshape(img_size[1],img_size[0])
plt.imshow(img)
plt.show()
'''

batch_size = 20
num_classes = 2
num_epochs = 20
num_channels = 1
num_filters = 4
num_pool = 2
kernel_size = 3

(X,y) = (train_data[0],train_data[1])
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
#proccess train and test input data
x_train = x_train.reshape(x_train.shape[0],img_size[1],img_size[0])
x_train = x_train.astype('float32')
x_train /= 225
x_test = x_test.reshape(x_test.shape[0],img_size[1],img_size[0])
x_test = x_test.astype('float32')
x_test /= 225

#y_train = np_utils.to_categorical(y_train,num_classes)
#y_test = np_utils.to_categorical(y_test,num_classes)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=((img_size[1],img_size[0]))))
model.add(keras.layers.Dense(256, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))

optimizer = tf.train.RMSPropOptimizer(0.001)
model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])

model.fit(x_train, y_train, epochs=num_epochs)
print('train:',model.evaluate(x_train, y_train))
print('test:',model.evaluate(x_test, y_test))
model.save('v0.1')
#plot

def plot_image(i, predictions_array, true_label, img):
	prediction, true_label, img = predictions_array[i], true_label[i],\
	 img[i].reshape((img_size[1],img_size[0]))
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img)
	plt.xlabel("{} ({})".format(prediction[0],true_label),
                                color='black')
	plt.show()

predictions = model.predict(x_test)
for i in range(99):
	plot_image(i, predictions, y_test, x_test)