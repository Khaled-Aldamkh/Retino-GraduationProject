# Retino Graduation Project
Graduation project for bachelor degree in the field of computer engineering at king Saud university
### Students: : </br>
 Khaled Aldamkh </br>
 Odai Alageel </br>
 Mohammed Almansour </br>


### Supervisor: 
 Dr.Yakoub Bazi


# About the project 
Diabetic Retinopathy (DR) is one of the highest reasons for blindness. It is hard to detect by just looking at the fundus images and there is a high chance of human error in grading the severity. In this report, we present an automatic solution based on Convolution neural networks (CNNs) for grading the severity of DR consisting of five severity stages. In the experiments, we consider the APTOS dataset, which contains 3662 fundus color images. In this phase of the project, we have designed a simple model from scratch. Also, we have designed an approach based on light-weight models called MobileNet and implement it on Android mobile device

<img width="1293" alt="Screen Shot 2021-12-21 at 1 06 12 PM" src="https://user-images.githubusercontent.com/83476407/146911535-5fb683e5-cf99-4b4a-9996-cb2454198fca.png">

# Table Of Contents
## Design Approaches 
Two approaches have been conducting. The first one we have designed a CNN from scratch, and the second one was to use the concept of transfer learning in which we take an existing model, and we fine tune it to our problem
### 1- CNN from scratch 
we start building our conventional neural network (CNN) as shown in the figure bellow our approach in designing our CNN was good, we got a fine accuracy however using small dataset has been proven inefficient since CNNs need large number of iterations before we find the optimal values, therefore if we have small dataset and large number of iterations an overfitting may occur. To solve this problem, we will design our model using transfer learning we will take an existing model called MobileNetV2 which have a little parameter to fit in a mobile application and we will use it in this project.
![scratch](https://user-images.githubusercontent.com/96137439/146664878-c9c58859-593f-44ce-8cb7-7697c518fd06.jpg)
### 2- CNN from transfer learning (MobileNetV2)
Transfer learning: Usually when we want to train a CNN we need a large dataset for better performance, because Messidor and APTOS datasets do not consider as a large datasets we used the concept of transfer learning which uses trained model and fit it to our problem which is effective for little data since Messidor dataset is just 1200 images and it will give us better performance when we consider the training time which we also need that for our new dataset (Asia Tele-Ophthalmology Society (APTOS)), so it will save training time because training a model from scratch can be exhausting and it is computationally intensive. So, we will design our model using transfer learning, we searched for a lightweight model and found a model called MobileNetV2 which have little parameters compared to the others
![image](https://user-images.githubusercontent.com/96137439/146664870-b330ad32-8846-4d04-a8a8-45268bce179f.png)
## Implementation the model on android app
 After we finished building our model, we want to implemntet on android app, so that to be in reach hands to everyone.
 There are two away to classify DR image 
 * Taking photos by mobile camera
 * Uploading image from mobile device
 ## Dataset
 In this project we used APTOS dataset provided by kaggle 
 https://www.kaggle.com/c/aptos2019-blindness-detection
