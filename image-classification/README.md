Tutorial - Image classification with descriptors + classifiers
===============================================================

Introduction
============

The problem of image classification can be hard. If we give a matrix of RGB elements to most of the methods of Machine Learning (SVM, Random Forest, KNN) the accuracy of the classification will hardly be good (except for some neural networks).

That's why we use descriptors. A descriptor take in input the image and returns a vector (or a matrix, it depends on the descriptor) which includes informations about the given image. We will use the outputs of descriptors to realize image classification.


Datasets
========

Road signs
----------

The dataset that we will use is a bank of road signs. The labels of these images are the type of road signs. There are 5 types (in french) : interdiction, obligation, priorite, vitesse and danger.
Dataset extracted from http://benchmark.ini.rub.de/?section=gtsrb&subsection=news.

Short explanation of the source code
=====================================

Data loading
-------------

We have files labels\_train.csv and labels\_test.csv which include the name of each picture and the labels. We will load the images and store the descriptors in the variables X\_train and X\_test. After that, **we will only work with these descriptors**. In this tuto, we will use the HOG descriptor.

Descriptor and classifier
-------------------------

We have chosen a HOG descriptor and a SVM classifier. You can find their parameters at the beginning of the source code (train\_test.js). After running the code, you'll see that the accuracy is 100% with SVM (NU SVC with a polynomial kernel of degree 3). 
HOG descriptor isn't the only descriptor that we can use. It can be interesting to try with descriptors like SIFT or SURF. Moreover, you can try others methods of Machine Learning. You can try this classification with a KNN classifier, or a Random Forest.

Tuning the source code
----------------------

You can easily change the classifiers and the descriptor in this source code. To change the classifier, modify the following line and modify the content of the variable options if needed :
```
var classifier = new SVM(options);
```


To change the classifier, you will have to change the following line (which is included in the two functions to load data) :
```
var descriptor = hog.extractHOG(image, options_hog);
```


Installation and running
=========================

The library for the HOG descriptor is not already in the NPM, so I've included it in this repo. To install everything, you have to do :
```
cd hog
npm install
cd ..
npm install
```
Then, you are ready to run the code by using the command-line **node index.js**.


Conclusion
==========

With this dataset, the combinaison of HOG and SVM works fine ! However we have to precise that recognition of road signs isn't really hard. 
You can easily use the source code with another dataset, provided you transform this dataset into the right format. A common exercice in image classification is the [CIFAR 10](https://www.cs.toronto.edu/~kriz/cifar.html).
The library of the HOG descriptor will soon be available in the NPM. You can find the repository in the project [image-js](https://github.com/image-js/hog). The library used for the SVM is a part of the [mljs project](https://github.com/mljs). You can try other classifier instead of SVM.
