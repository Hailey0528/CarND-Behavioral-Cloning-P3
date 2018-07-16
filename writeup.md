# **Behavioral Cloning Project**

#### This project is using convolutional neural network to predict the steering angle of a car. The CNN architecture from NVIDIA for End to End Learning for Self-Driving Cars is the initial architecture. Relu and Dropout are added to avoid the overfitting. With the trained model the car can drive well in the Udacity Simulator.


[//]: # (Image References)


[image1]: ./image/CNN_Architecture.png "NVIDIA.jpg"
[image2]: ./image/flipping.png "flipping"
[image3]: ./image/Preprocessing.png "Preprocessing"
[image4]: ./image/Angle_Distribution.jpg "Angle_Distribution.jpg"


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used CNN architecture of NVIDIA for End to End Learning for Self-Driving Cars. The input of the model is The model consists of a convolution neural network with 5x5, 3x3 filter sizes and depths between 24 and 64 (model.py lines 18-24) 
![alt text][image1]

The model includes RELU layers to introduce nonlinearity (code line 139-155), and the data is normalized in the model using a Keras lambda layer (code line 138). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 158). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 161). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 161).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I only used center lane driving data, which can already make the car drive very well on the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
I added relus, weights regularizer, bias regularizer in every layer of the NVIDIA architecture. Then dropout is added to avoid overfitting. I tried to add drop in every layer and add it only in the last fully connected layer. It shows that the loss is lower with only dropout in the last fully connected layer, which I have tried in the last project Traffic Sign Classification. 

I choose validation split with 0.2 in order to let the validation result to check the model. After trying to tune the parameter of weights and bias regularizer and choosing the dropout layer, I run the model for 50 epochs in order to find the optimal epochs number.

The final step was to run the simulator to see how well the car was driving around track one. The car was driving very well on the track even when the set speed is 25 km/h without leaving the road. 

#### 3. Creation of the Training Set & Training Process

I used the data set of Udacity at first. I also tried to get data with the simulator. But since the car drives very well, there is no need to add more images. The images from left and right camera are also used in order to increase the data number. To augment the data sat, I also flipped images and angles thinking that this would add more steering angle data. For example, here is an image that has then been flipped:
![alt text][image2]

After the collection process, I had 48216 number of data points. This is the distribution of steering angles of the data.
![alt text][image4]

I then preprocessed this data by cropping, resizing each image and converting the image from BGR to YUV, because the input of architecture from NVIDIA is YUV image. Here is a example of image which has been preprocessed.
![alt text][image3]

I finally randomly shuffled the data set and put 20% of the data into a validation set. With the initial model in 5 Epochs I got training loss with 0.0230 and validation loss 0.0254. Then I tried to add dropout in every layer, after 5 epochs, the training loss is 0.0226 and the validation loss is 0.0242. The validation loss is higher than the training loss, which means there is overfitting. Then I tried to add a dropout layer of 20% in the last fully connected layer. After 5 epochs, the training loss is 0.0255 and the validation loss is 0.0238. I also changed the Sigma from 0.001 to 0.005, the training and validation loss becomes double as before.Therefore, I chose sigma with 0.001, and just add dropout in the last fully connected layer.

I used this training data for training the model. Then I tried 50 epochs with this architecture, then I found in the 17th epoch the result is the best. I used an adam optimizer so that manually training the learning rate wasn't necessary.
