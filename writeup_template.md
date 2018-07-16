# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./image/Angle_Distribution.jpg "Angle_Distribution.jpg"
[image5]: ./image/flipping.jpg "flipping"
[image6]: ./image/Preprocessing.jpg "Preprocessing"


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used CNN architecture of NVIDIA for End to End Learning for Self-Driving Cars. The input of the model is The model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.
I used the data set of Udacity at first. The images from left and right camera are also used in order to increase the data number. To augment the data sat, I also flipped images and angles thinking that this would add more steering angle data. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 48216 number of data points. I then preprocessed this data by cropping, resizing each image and converting the image from BGR to YUV, because the input of architecture from NVIDIA is YUV image. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 
Batch size is 256, 
With the initial model in 5 Epochs I got training loss with 0.0230 and validation loss 0.0254. I tried to add a dropout layer of 20% in the last fully connected layer. After 5 epochs, the training loss is 0.0255 and the validation loss is 0.0238. Then I tried to add dropout in every layer, after 5 epochs, the training loss is 0.0226 and the validation loss is 0.0242. I also changed the Sigma from 0.001 to 0.005, the training and validation loss becomes double as before.Therefore, I chose sigma with 0.001, and just add dropout in the last fully connected layer.
Then I tried to add some data which are by myself.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Then I tried 50 epochs with this architecture, then I found in the 31th epoch the result is good. I used an adam optimizer so that manually training the learning rate wasn't necessary.
