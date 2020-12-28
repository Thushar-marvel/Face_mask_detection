# Face_mask_detection
In this project, we will train a face mask detector with OpenCV, Keras/TensorFlow, and Deep Learning.

# Steps involved in building face mask detector model
The model is trained  in jupyter notebook
1.Load packages: Numpy to work with arrays,os and glob to load folders, Keras with tensorflow as backend for training the model, Matplotlib for plotting graphs and sklearn for model evaluation

2.Loading data - The dataset used in the project consists of 1915 with mask and 1918 with out mask images.
The dataset can be downloaded from https://github.com/prajnasb/observations/tree/master/experiements/data github portal

3.Data preprocessing: Before feeding to the model the data should be preprocessed to best fit model.The images are converted to array and these are normalized by dividing with 255.Random flipping , shrinking of the images are done for better accuracy.

4.Model building: Using keras , model is build  3 convolution layer,3 pooling layer,2 Fully connected layer , 1 dropout layer and final output layer.Relu activation function is used in hidden layer and softmax in output layer.

5.Model compile and training: The model is trained for 20 epochs with adam optimizer and binary_crossentropy loss function.

6.Model evaluation: The developed model got an training accuracy of 97.8% and test accuracy of 97.5%

# Steps involved in real time face mask detector using open cv:
1.Load pacakages: Along with above loaded package opencv is installed.

2.Load pre-trained model: We have used pre-trained caffemodel to detect faces in the frames/images and apply those detected face images to the above trained classification model to detect mask in the image 

3.Mark bounding box around the detected face image: Caffemodel with provide the co ordinates of the face image in the full image. These coordinates are used to mark the bounding box around the face along with confidence level from the classification model.
