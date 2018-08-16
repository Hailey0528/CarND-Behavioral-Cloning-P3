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
data = pandas.read_csv('../../ubuntu/data/driving_log.csv', skiprows=[0], skipinitialspace=True,names=columns)
center = data.center.tolist()
center_recover = data.center.tolist() 
left = data.left.tolist()
right = data.right.tolist()
steering = data.steering.tolist()
steering_recover = data.steering.tolist()

#  Shuffle center and steering. Use 10% of central images and steering angles for validation.
center, steering = shuffle(center, steering)
center, X_valid, steering, y_valid = train_test_split(center, steering, test_size = 0.15, random_state = 100) 
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
    image = cv2.imread('../../ubuntu/data/'+img_train[i])   
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

# Generate random brightness function, produce darker transformation 
def random_brightness(image):
    #Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #Generate new random brightness
    rand = random.uniform(0.3,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    #Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img 

## generate data
def generator(batch_size):
    batch_train = np.zeros((batch_size, 66, 220, 3), dtype=np.float32)
    batch_angle = np.zeros((batch_size), dtype=np.float32)
    while True:
        data, angle = shuffle(X_train, y_train)
        for i in range(batch_size):
              # random choose one image in train data and read it
              index = int(np.random.choice(len(data), 1))
              img=cv2.imread(data[index].strip())
              batch_train[i] = resizing(cropping(random_brightness(img)))
              batch_angle[i] = angle[index]#*(1+np.random.uniform(-0.10,0.10))
              flip_coin = random.randint(0, 1)
              if flip_coin==1:
                 batch_train[i] = np.fliplr(batch_train[i])
                 batch_angle[i] = -batch_angle[i]
        yield batch_train, batch_angle
	
def generator_valid(batch_size):
    batch_valid = np.zeros((batch_size, 66, 220, 3), dtype=np.float32)
    batch_angle = np.zeros((batch_size), dtype=np.float32)
    while True:
        data, angle = shuffle(X_valid, y_valid)
        for i in range(batch_size):
              # random choose one image in train data and read it
              index = int(np.random.choice(len(data), 1))
              img=cv2.imread(data[index].strip())
              batch_valid[i] = resizing(cropping(img))
              batch_angle[i] = angle[index]#*(1+np.random.uniform(-0.10,0.10))
        yield batch_valid, batch_angle
	
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
#X_train = np.array(preprocessing(X_train))
#y_train = np.array(y_train)

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
model.fit_generator(generator(batch_size), samples_per_epoch = math.ceil(len(X_train)), nb_epoch=2, validation_data = generator_valid(batch_size), nb_val_samples = len(X_valid))


model.save('model.h5')
