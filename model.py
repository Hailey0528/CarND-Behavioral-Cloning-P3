import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.regularizers import l2, activity_l2
from PIL import Image
lines = []
with open('../../ubuntu/data/data/driving_log.csv') as csvfile:
	next(csvfile, None)
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
#preprocessing of the images
def preprocessing(img, number_top, number_bottom, number_left, number_right):
	#cropping the image for important informations
	size = img.shape
	img_crop = img[number_top:(size[0]-number_bottom), number_left:(size[1]-number_right)]
	
	#resize the image
	new_height = 66
	new_width = 220
	img_resize = img.resize((new_width,new_height), Image.ANTIALIAS)
	
	return img_resize
 
images = []
measurements = []
for line in lines:
        #steering angle
	angle = float(line[3])
	#set delta angle for the left and right camera
	delta_angle = float(0.25)

	#####for the central camera##### 	
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '../../ubuntu/data/data/IMG/'+filename
	image_central = cv2.imread(current_path)
	#image_central = np.array(image_central)
	image_central = preprocessing(image_central, 50, 20, 0, 0)
	images.append(image_central)
	measurements.append(angle)
	#flip the image and save the image and relative steering angle
	#image_central_flip = cv2.flip(image_central, 1))
	images.append(preprocessing(cv2.flip(image_central, 1)))
	measurements.append(angle*-1.0)

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




#get the input train and validation data, and the expected output of the model
X_train = np.array(images)
print(X_train.shape)
y_train = np.array(measurements)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda, Cropping2D, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(3, 160, 320)))
model.add(Reshape((66, 220, 3), input_shape=(90, 320, 3)))
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
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, batch_size=32)

model.save('model.h5')
