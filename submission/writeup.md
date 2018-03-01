# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The steps of this project are the following:
* Load the data set which is provided in the classroom, which is a resized partial [German Traffic Sign data set](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* Explore, summarize and visualize the data set
* Pre-process origin data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./figs/label_name_table.PNG "Labels"
[image2]: ./figs/original.PNG "Original Sign"
[image3]: ./figs/scaled.PNG "Scaled"
[image4]: ./figs/rotated.PNG "Rotated"
[image5]: ./figs/after_gen_distribution.PNG "Label Distributions Comparison"
[image6]: ./figs/origin_vs_preprocessed.PNG "Original vs Pre-processed"
[image7]: ./figs/web_imgs.PNG "Web Images"
[image8]: ./figs/preprocessed_web_imgs.PNG "Pre-processed Web Images"
[image9]: ./figs/predict_results.PNG "Prediction Results"
[image10]: ./figs/top_5_softmax_result.PNG "Top 5 Possible Results"
[image11]: ./figs/label_distribution.PNG "Label Distributions"
[image12]: ./figs/nn_structure.PNG "Neural Network Structure"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/JayLSU/CarND-Traffic-Sign-Classifier-Project/tree/master/submission/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32,32,3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here are some exploratory visualizations of the data set. The first is the table for label class ID and its corresponding name.

![alt text][image1]

Then I used a bar figure to show the distribution of labels in trainning set.

![alt text][image11]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In data pre-processing, I decided to convert the images to grayscale because I can focus on shape features on traffic signs which is more important than color. Additionally, I can reduce the number of weights and speed up the training and learning process. After grayscaled the images, I normalized the data to ensure that each feature to have a similar range and therefore one global learning rate is good enough. Since all features are in similar range, the weights would be in similar range too.

Here is an example of traffic sign images before and after grayscaling and normalization.

![alt text][image6]

As you can observe in the distribution figure, some label classes are quite few compared with others, i.e., class 0. This may add some bias to the network and result in a bad performance for predicting some signs. Therefore, I decided to add some augmented images to the sign images whose number are less than 750. In augmented image generating process, I used 'scaling' and 'rotating' to the original image. One example is shown as following.

![alt text][image2] ![alt text][image3] ![alt text][image4]

After added augmented images, the distribution becomes more better.

![alt text][image5]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I first used the LeNet with a dropout in my model and achieved around 94% validation accurancy. Then to further improve the accurancy, I added a 3x3 convolutional (conv) layer after the 2nd conv layer in the LeNet, then concatenated the added conv layer with the 2nd conv layer. After the first fully connected layer, I also added a droppout layer. The rest structures are the same with the LeNet. The validation accurancy achieved around 96%. Inspired by inception structure, I added more conv layers and concatenated them together. Finally, I obtained the validation accurancy around 97%. My final model is constructed as following.

![alt text][image12]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an EPOCHS = 50, BATCH_SIZE = 256, and learning rate = 0.001. AdamOptimizer is used.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.975.
* test set accuracy of 0.948.

The original LeNet with dropout layer can satisfy the requirment, however, I still want to improve it. Since I think MNIST images have less features than traffic signs. I decided to first add one more conv layer. Because concatenating layers into one can provide more details for the next layer, I concatenated the added layer with the 2nd conv layer in LeNet. To further improve the accurancy, in my final model, I used an Inception v4 similar structure. I think probably because more layers in Inception v4 structure can exploit more details of the image, the final accurancy is better. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I finally used 9 images that I found on the web:

![alt text][image7] 

The pre-processed images are:

![alt text][image8] 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image9] 

The model was able to correctly guess 5 of the 9 traffic signs, which gives an accuracy of 0.556. Much worse than test set accuracy. Probably the web images have lower resolution (image 8), more complexed background (image 6) and are pictured from a different angle (image 9). The prediction results are not as good as test set. From the prediction result, I believe I need to read some peer's paper to further improve my neural network.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

The top 5 probabilities of each web image prediciton are shown below.

![alt text][image10] 