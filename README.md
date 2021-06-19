# OBJECT DETECTION USING DEEP LEARNING 

## A. PROJECT SUMMARY

**Project Title:** OBJECT DETECTION USING DEEP LEARNING

**Team Members:** 
- MD SHADMAN ZOHA
- Nafish Ahammed
- ISRAT JAHAN
- Israt Jahan Bhuiyan


 **Objectives:**
- To build a system that can guess the name of an object from film or real-time footage. 



##  B. ABSTRACT 

Humans can detect and identify objects present in an image. The human visual system is fast and accurate and can also perform complex tasks like identifying multiple objects and detect obstacles with little conscious thought. The availability of large sets of data, faster GPUs, and better algorithms, we can now easily train computers to detect and classify multiple objects within an image with high accuracy. We need to understand terms such as object detection, object localization, loss function for object detection and localization. Object detection is the method of detecting real-world object instances in still images or videos, such as a vehicle, bike, TV, flowers, and people. It helps us to recognize, localize, and detect different objects within an image, allowing us to have a greater understanding of the image as a whole. It is widely used in image retrieval, protection, surveillance, and a variety of other applications. Image classification involves activities such as predicting the class of one object in an image. Object localization is refers to identifying the location of one or more objects in an image and drawing an abounding box around their extent. Object detection does the work of combines these two tasks and localizes and classifies one or more objects in an image. Traditional object detection methods are built on handcrafted features and shallow trainable architectures. Their performance easily stagnates by constructing complex ensembles which combine multiple low-level image features with high-level context from object detectors and scene classifiers. With the rapid development in deep learning, more powerful tools, which are able to learn semantic, high-level, deeper features, are introduced to address the problems existing in traditional architectures.
Object detection can be accomplished in a variety of ways:
- Detection of Objects Using Features
- Object Detection SVM Classifications with HOG Features by Viola Jones
- Object Detection Using Deep Learning


![Screenshot (102)](https://user-images.githubusercontent.com/80869517/122650576-92c8f180-d155-11eb-9c29-6117ca4065f5.png)



## C.  DATASET

This project is an attempt to demonstrate about how we can use object detector to recognize any object, location of an object in any image and also count the number of instances of an object.
For this project we are going to use COCO dataset, which represents - Common Objects in Context. The COCO dataset provides a base dataset and benchmark for evaluating the periodic improvement of the models through computer vision research. Coco has some other features which are object segmentation, recognition in context, superpixel stuff segmentation, 1.5 million object instances, 80 object categories .  
In object detection for input an image can be taken with one or more pictures such as a picture and for output we will get one or more bounding boxes and a class label for each bounding box.


![image](https://user-images.githubusercontent.com/80869517/122103283-44a3ad80-ce38-11eb-8e03-edd7ca140390.png)


Figure 2 shows the proccess of dataset.


![Screenshot (88)](https://user-images.githubusercontent.com/80869517/122104550-97319980-ce39-11eb-9b07-6a854a034008.png)

Figure 3 shows the example of dataset.

## D.   PROJECT STRUCTURE


The following directory is our structure of our project:
$ tree --dirsfirst --filelimit 15 .
- │ ├── numpy as venv
- │ └── object_detection
- ├── cv2
- │ ├── cv2.VideoCapture(0)
- │ ├── fr.load_image_file("car.jpg")
- │ └── fr.load_image_file("tv.jpg")
- ├── dataset
- │ ├── coco.names
- ├── frozen_inference_graph.pb
- ├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
- └── cv2.rectangle
- └── video_capture.release()
- 10 files





## E.   TRAINING AND CONFIGURE THE OBJECT DETECTION

- We can now use jupyter nootbook and Tensorflow to train our model.

We go to the project folder, which contains the files. The model will then be pre-processed, trained, and tested in a notebook. To begin, we must name our classes. So, using this command, we make a file and put the coco.names in it:

classNames= []

classFile = 'coco.names.txt'

with open(classFile,'rt') as f:

classNames = f.read().rstrip('\n').split('\n')

    
- As a result, all of our classes are now set. Now it's time to build our project. The most often used mobilenet for object detection is ssd mobilenet v3. So, to detect our object, we're utilizing ssd mobilenet v3 and frozen inference graph. We are configuring our project using this command:


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

weightsPath = 'frozen_inference_graph.pb'


## F.  RESULT AND CONCLUSION

- Detecting object in real time with TensorFlow.

Object detection is a key ability for most computer and robot vision system.
Due to its powerful learning ability and advantages in detecting any objects within seconds, deep learning based object detection has been a research hotspot in recent years. Although great progress has been observed in the last years, we are still far from achieving human-level performance, in particular in terms of open-world learning. It is a true fact that object detection has not been used much in many areas where it could be of great help. As mobile robots, and in general autonomous machines, are starting to be more widely deployed, the need of object detection systems is gaining more importance. 

Finally, we need to consider that we will need object detection systems for nano-robots or for robots that will explore areas that have not been seen by humans, such as depth parts of the sea or other planets, and the detection systems will have to learn to new object classes as they are encountered. In such cases, a real-time open-world learning ability will be critical.

- Output of result shown below:

https://user-images.githubusercontent.com/80869517/122649391-69a56280-d14f-11eb-99c4-6cded3cc7fa3.mp4



## G.   PROJECT PRESENTATION 

- Coming Soon




