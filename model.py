import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image

lines = []
### get the informations from csv file of data of Udacity
with open('../../ubuntu/driving_log.csv') as csvfile:
    next(csvfile, None)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
# print the number of the lines
print(len(lines))

# calculate the number of the images in the Undacity data
length_original = len(lines)

### get the informations from csv file of added data 
with open('../../ubuntu/added_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# print the number of the total lines
print(len(lines))

### read the images with the folderName and the information 
### for example, the name of image from the central camera, left camera, right camera, the sterring angle. 
### read every image and flip the image, and 
def image_flip(line, folderName):
	delta_angle = 0.2
	angle = float(line[3])

	#####for the central camera#####
	source_path = line[0]
	# get the image name from the csv file
	filename = source_path.split('/')[-1]
	# create the path of the image
	current_path = folderName + filename
	# read the image
	image = cv2.imread(current_path)
	# If there is image with this name, then append the image data to the image array and the steering angle to the steering angle array
	# flip this image and steering angle, then append the flipped image into the image array and corresponding steering angle to the steering angle array
	if image is not None:			
		# save the image data from central camera and the steering angle
		images.append(image)	
		measurements.append(angle)
		# flip the image from central camera
		image_flip = np.fliplr(image)
		# save the flipped image data and the corresponding steering angle
		images.append(image_flip)
		measurements.append(-angle)

	#####for the left camera#####
	source_path = line[1] 
	# get the image name from the csv file
	filename = source_path.split('/')[-1]
        # create the path of the image
	current_path = folderName + filename
	# read the image	
	image_left = cv2.imread(current_path)
	# If there is image with this name, then append the image data to the image array and the steering angle to the steering angle array
        # flip this image and steering angle, then append the flipped image into the image array and corresponding steering angle to the steering angle array
	if image_left is not None:
                # save the image data from left camera and the steering angle
		images.append(image_left)
		measurements.append(angle+delta_angle)
		#flip the image from left camera and save the image and corresponding steering angle
		images.append(np.fliplr(image_left))
		measurements.append(-delta_angle-angle)

	#####for the right camera#####
	source_path = line[2] 
	# get the image name from the csv file
	filename = source_path.split('/')[-1]
	# create the path of the image
	current_path = folderName + filename
	# read the image
	image_right = cv2.imread(current_path)
        # If there is image with this name, then append the image data to the image array and the steering angle to the steering angle array
        # flip this image and steering angle, then append the flipped image into the image array and corresponding steering angle to the steering angle array
	if image_right is not None:		
		# save the image data from right camera and the steering angle
		images.append(image_right)
		measurements.append(angle-delta_angle)
		#flip the image from right camera and save the image and corresponding steering angle
		images.append(np.fliplr(image_right))
		measurements.append(-angle+delta_angle)

def image_flip_central(line, folderName):
	delta_angle = 0.2
	angle = float(line[3])

	#####for the central camera#####
	source_path = line[0]
	# get the image name from the csv file
	filename = source_path.split('/')[-1]
	# create the path of the image
	current_path = folderName + filename
	# read the image
	image = cv2.imread(current_path)
	# If there is image with this name, then append the image data to the image array and the steering angle to the steering angle array
	# flip this image and steering angle, then append the flipped image into the image array and corresponding steering angle to the steering angle array
	if image is not None:			
		# save the image data from central camera and the steering angle
		images.append(image)	
		measurements.append(angle)
		# flip the image from central camera
		image_flip = np.fliplr(image)
		# save the flipped image data and the corresponding steering angle
		images.append(image_flip)
		measurements.append(-angle)
		
images = []
measurements = []
# save the image data from Udacity and the steering angle, the flipped image of the original image should also be saved
for line in lines[:length_original]:
	image_flip(line, '../../ubuntu/IMG/')
print(len(images))

# save the image data with the simulator by myself and the steering angle, the flipped image of the original image should also be saved
for line in lines[length_original:]:
	image_flip_central(line, '../../ubuntu/added_data/IMG/')

print(len(images))

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
X_train = np.array(preprocessing(images))
y_train = np.array(measurements)

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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=17)

model.save('model.h5')
