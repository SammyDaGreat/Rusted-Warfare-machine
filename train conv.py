import os
print('Dataset size:',len(os.listdir('grayData\\x\\')))
import tensorflow as tf
import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from keras import utils as np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
import time

# Data info
path_x = 'grayData\\x\\'
path_y1 = 'grayData\\y1\\'
path_y2 = 'grayData\\y2\\'
img_size = [576,778]
num_samples = 30
num_sample_iterations = 90
input_shape=(img_size[1],img_size[0],1)
test_proportion = 0.04

# Training parameters
batch_size = 15
num_epochs = 50

# Plot function
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

# Set up model
print('Preparing CNN...\n')
model = keras.models.Sequential()
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),input_shape=input_shape))
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

optimizer = tf.train.RMSPropOptimizer(0.001)
model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])

# Prompt for model name
model_file_name = input('Enter model name to save:\t')
print('Starting to train!\n')

all_test_x = []
all_test_y = []
# Train the model
for i in range(0,num_sample_iterations):
        print('Preparing to start iteration '+str(i+1)+'...\n')
        time.sleep(1)
        
        # Read data
        imlist = os.listdir(path_x)[(i*num_samples):(i*num_samples+num_samples)]

        immatrix = np.array([np.load(path_x + file_name).flatten() for file_name in imlist],'f')
        label = np.array([np.load(path_y1 + file_name).flatten() for file_name in imlist],'f')

        data,Label = shuffle(immatrix,label,random_state = 2)
        train_data = [data,Label]

        (X,y) = (train_data[0],train_data[1])
        x_train,x_test,y_train,y_test = train_test_split(X, y, test_size = test_proportion, random_state = 4)
        try:
            all_test_x.append(x_test)
            all_test_y.append(y_test)
        except:
            pass
        # Process data
        x_train = x_train.reshape(x_train.shape[0],img_size[1],img_size[0],1)
        x_train = x_train.astype('float32')
        x_train /= 225
        x_test = x_test.reshape(x_test.shape[0],img_size[1],img_size[0],1)
        x_test = x_test.astype('float32')
        x_test /= 225

        # BP
        model.fit(x_train, y_train, epochs=num_epochs)
        
        print('Iteration '+str(i+1)+' training:',model.evaluate(x_train, y_train))
        print('Iteration '+str(i+1)+' test results:',model.evaluate(x_test, y_test))

        # Save model
        model.save(model_file_name)
        print('Saved model as \''+model_file_name+'\'')
        print('Iteration '+str(i+1)+' complete!\n|\n|\n|\n|\n|\n|\n|\n|\n|\n|\n|\n|\n|\n|\n|\n|\n|\n|\n|\n')
        time.sleep(3)

# Plot test result
print('all_test_x length:',len(all_test_x))
print('one element shape:',np.shape(all_test_x[0]))
predictions = model.predict(x_test)

for i in range(x_test.shape[0]):
	plot_image(i, predictions, y_test, x_test)
