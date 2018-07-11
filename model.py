import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

lines = []
with open('../../Documents/data/data/driving_log.csv') as csvfile:
    next(csvfile, None)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print(len(lines))

#get images
images = []
measurements = []
for line in lines:
	measurement = float(line[3])
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '../../Documents/data/data/IMG/'+filename
	image = cv2.imread(current_path)
	images.append(image)	
	measurements.append(measurement)
	image_flip = np.fliplr(image)
	images.append(image_flip)
	measurements.append(measurement*-1)


	#####for the left camera#####
	#source_path = line[1] 
	#filename = source_path.split('/')[-1]
	#current_path = '../../ubuntu/data/data/IMG/'+filename
	#image_left = cv2.imread(current_path)
	#image_left = np.array(image_left)
	#images.append(image_left)
	#measurements.append(angle+delta_angle)
	#flip the image and save the image and relative steering angle

	#images.append(cv2.flip(image_left, 1))
	#measurements.append(-delta_angle-angle)

	#####for the right camera#####
	#source_path = line[2] 
	#filename = source_path.split('/')[-1]
	#current_path = '../../ubuntu/data/data/IMG/'+filename
	#image_right = cv2.imread(current_path)
	#image_right = np.array(image_right)
	#images.append(image_right)
	#measurements.append(angle-delta_angle)
	#flip the image and save the image and relative steering angle
	#images.append(cv2.flip(image_right, 1))
	#measurements.append(-angle+delta_angle)
print(len(images))

def cropping(img):
	#cropping the image for important informations
	number_top = 50
	number_bottom = 20 
	number_left = 0 
	number_right = 0
	size = img.shape
	img_crop = img[number_top:(size[0]-number_bottom), number_left:(size[1]-number_right)]
	return img_crop

def resizing(img):
	#resize the image
	new_height = 66
	new_width = 220
	img_resize = cv2.resize(img, (new_width,new_height))
	return img_resize

def preprocessing(img):
	image_Preprocessing = []
	n_training = len(img)
	for i in range(n_training):
		image = cropping(img[i])
		image = resizing(image)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
		image_Preprocessing.append(image)		
	return image_Preprocessing

#get the input train and validation data, and the expected output of the model
X_train = np.array(preprocessing(images))
print(X_train.shape)
y_train = np.array(measurements)


# model architecture
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda
from keras.regularizers import l2, activity_l2
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:x/255.0-0.5, input_shape=(66, 220, 3)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.01), activation='relu'))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.01), activation='relu'))
#model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.01), activation='relu'))
#model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.01), activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.01), activation='relu'))
model.add(Flatten())
model.add(Dense(100), W_regularizer=l2(0.01))
model.add(Dense(50), W_regularizer=l2(0.01))
model.add(Dense(10), W_regularizer=l2(0.01))
model.add(Dense(1))
#model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, batch_size=32)

model.save('model.h5')
