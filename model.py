import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image
import pandas
import random
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Read the columns from driving_log.csv 
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv('../../Documents/data/driving_log.csv', skiprows=[0], skipinitialspace=True,names=columns)
center = data.center.tolist()
center_recover = data.center.tolist() 
left = data.left.tolist()
right = data.right.tolist()
steering = data.steering.tolist()
steering_recover = data.steering.tolist()

#  Shuffle center and steering. Use 10% of central images and steering angles for validation.
center, steering = shuffle(center, steering)
center, X_valid, steering, y_valid = train_test_split(center, steering, test_size = 0.10, random_state = 100) 
# Divide the training data into 3 groups based on the steering angle >0.15, <-0.15, or between -0.15 and 0.15
img_center, img_left, img_right = [], [], []
steer_center, steer_left, steer_right = [], [], []
for i in steering:
  index=steering.index(i)
  if i > 0.15:
    img_right.append(center[index])
    steer_right.append(i)
  if i < -0.15:
    img_left.append(center[index])
    steer_left.append(i)
  else:
    img_center.append(center[index])
    steer_center.append(i)

# add recovery to make the number of straight images, left images and right images are equal
center_size, left_size, right_size = len(img_center), len(img_left),len(img_right)
main_size = math.ceil(len(center_recover))
l_add = center_size-left_size
r_add = center_size-right_size

# Generate random list of indices for left and right recovery images
indice_L = random.sample(range(main_size), l_add)
indice_R = random.sample(range(main_size), r_add)
#print(indice_L, indice_R)
delta_angle = 0.2
# Filter angle less than -0.15 and add right camera images into driving left list, minus an adjustment angle #
for i in indice_L:
  if steering_recover[i] < -0.15:
    img_left.append(right[i])
    steer_left.append(steering_recover[i] - delta_angle)

# Filter angle more than 0.15 and add left camera images into driving right list, add an adjustment angle #  
for i in indice_R:
  if steering_recover[i] > 0.15:
    img_right.append(left[i])
    steer_right.append(steering_recover[i] + delta_angle)

## COMBINE TRAINING IMAGE NAMES AND ANGLES INTO X_train and y_train ##  
img_train = img_center + img_left + img_right
steering_train = np.float32(steer_center + steer_left + steer_right)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


batch_size = 128
num_classes = 10
epochs = 20

X_train = []
y_train = []

for i in range(len(img_train)):
    image = cv2.imread('../../Documents/data/'+img_train[i])   
    X_train.append(image)
    y_train.append(steering_train[i])
    #X_train.append(np.fliplr(image))
    #y_train.append(-steering_train[i]) 

def cropping(img):
	#cropping the image for important informations with the number of pixels, which should be cropped in every side
	number_top = 50
	number_bottom = 20 
	number_left = 0 
	number_right = 0
	size = img.shape
	img_crop = img[number_top:(size[0]-number_bottom), number_left:(size[1]-number_right)]
	return img_crop

def resizing(img):
	#resize the image, with new size (new_width, new_height)
	new_height = 66
	new_width = 220
	img_resize = cv2.resize(img, (new_width,new_height))
	return img_resize

def preprocessing(img):
	image_Preprocessing = []
	n_training = len(img)
	for i in range(n_training):
		# for every image, crop the iamge
		image = cropping(img[i])
		# resize the image
		image = resizing(image)
		# change the BGR image to YUV image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
		# save the image to the array for image data set
		image_Preprocessing.append(image)		
	return image_Preprocessing

#get the input train and validation data, and the expected output of the model
X_train = np.array(preprocessing(X_train))
y_train = np.array(y_train)

##### show visualzition of data, preprocessing #####
#show the distribution of training data
Angle_Min = min(y_train)
Angle_Max = max(y_train)
iteration = 0.04 
Angle_range = np.arange(Angle_Min, Angle_Max+iteration, iteration)
Length_range = len(Angle_range)
Number_eachRange = np.zeros(Length_range-1)
for angle in y_train:
	for i in range(Length_range-1):
		if angle >= Angle_range[i] and angle<=Angle_range[i+1]:
			Number_eachRange[i] += 1

# show the distribution of classes in the training
Distribution = plt.figure()
plt.bar(Angle_range[1:]*25, Number_eachRange)
plt.title('Distribution')
plt.xlabel('number of class')
plt.ylabel('number of images')
Distribution.savefig('image/Angle_Distribution.jpg')


## generate data
def generator(batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3))
    while True:
          data, angle = shuffle(X_train, y_train)
          for i in range(batch_size):
              index = int(np.random.choice(len(data), 1))
              batch_train[i] = 
    image = cv2.flip(image, 1)
##### model architecture#####
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda
from keras.regularizers import l2, activity_l2
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

sigma = 0.001
rate_dropout = 0.2
model = Sequential()
model.add(Lambda(lambda x:x/255.0-0.5, input_shape=(66, 220, 3)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(sigma), b_regularizer=l2(sigma), activation='relu'))
#model.add(Dropout(rate_dropout))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(sigma), b_regularizer=l2(sigma), activation='relu'))
#model.add(Dropout(rate_dropout))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(sigma), b_regularizer=l2(sigma), activation='relu'))
#model.add(Dropout(rate_dropout))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', W_regularizer=l2(sigma),b_regularizer=l2(sigma),  activation='relu'))
#model.add(Dropout(rate_dropout))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', W_regularizer=l2(sigma),b_regularizer=l2(sigma),  activation='relu'))
#model.add(Dropout(rate_dropout))
model.add(Flatten())
#model.add(Dropout(rate_dropout))
model.add(Dense(100, W_regularizer=l2(sigma), b_regularizer=l2(sigma), activation='relu'))
#model.add(Dropout(rate_dropout))
model.add(Dense(50, W_regularizer=l2(sigma), b_regularizer=l2(sigma), activation='relu'))
#model.add(Dropout(rate_dropout))
model.add(Dense(10, W_regularizer=l2(sigma), b_regularizer=l2(sigma), activation='relu'))
model.add(Dropout(rate_dropout))
model.add(Dense(1))

##### compile #####
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=17)

model.save('model.h5')
