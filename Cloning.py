import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []
with open('../../ubuntu/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = '../../ubuntu/IMG/'+filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)

#get the input train and validation data, and the expected output of the model
X_train = np.array(images)
print(X_train.shape)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:x/255.0-0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(32, 5, 5, activation='relu', input_shape=(160, 320, 3)))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Convolution2D(50, 5, 5, activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(60, 5, 5, activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())

model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))
#model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, batch_size=32)

model.save('model.h5')
