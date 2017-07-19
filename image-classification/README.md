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
Here are the options of the HOG descriptor that we will use :
```javascript
let options_hog = {
    cellSize: 4,
    blockSize: 2,
    blockStride: 1,
    bins: 6,
    norm: "L2"
}
```

We will load the data with the following code :
```javascript
let X_train = [];
let Y_train = [];
let X_test = [];
let Y_test = [];

async function loadData(){
    // We will load the dataset

    async function loadTrainingSet(){
        var lines = fs.readFileSync('labels_train.csv').toString().split("\n");
        for(var i = 0; i < lines.length; i++){
            var elements = lines[i].split(";");
            if(elements.length < 2)
                continue;
            var file = __dirname + "/data/" + elements[0];
            // in the variable X, we will store the HOG of the pictures
            var image = await Image.load(file);
            image = await image.scale({width:100, height:100});
            var descriptor = hog.extractHOG(image, options_hog);
            X_train.push(descriptor);
            Y_train.push(elements[1]);
        }
    }

    async function loadTestSet(){
        var lines = fs.readFileSync('labels_test.csv').toString().split("\n");
        for(var i = 0; i < lines.length; i++){
            var elements = lines[i].split(";");
            if(elements.length < 2)
                continue;
            var file = __dirname + "/data/" + elements[0];
	    // in the variable X, we will store the HOG of the pictures
            var image = await Image.load(file);
            image = await image.scale({width:100, height:100});
            var descriptor = hog.extractHOG(image, options_hog);
            X_test.push(descriptor);
            Y_test.push(elements[1]);
        }
    }

    await loadTrainingSet();
    await loadTestSet();
}
```

Descriptor and classifier
-------------------------

We have chosen a HOG descriptor and a SVM classifier. You can find their parameters at the beginning of the source code (train\_test.js). After running the code, you'll see that the accuracy is 100% with SVM (NU SVC with a polynomial kernel of degree 3). 
HOG descriptor isn't the only descriptor that we can use. It can be interesting to try with descriptors like SIFT or SURF. Moreover, you can try others methods of Machine Learning. You can try this classification with a KNN classifier, or a Random Forest.
The source code which corresponds to the SVM, its training and the predictions that it does is the following :
```javascript
let options = {
    type: SVM.SVM_TYPES.NU_SVC, 
    kernel : SVM.KERNEL_TYPES.POLYNOMIAL,
    degree : 3,
    nu : 0.1,
    shrinking : false
}

loadData().then(function(){
    // Now, the dataset should be loaded. We will apply the classification
    // Begin of the classification

    var classifier = new SVM(options);

    classifier.train(X_train, Y_train);
    test();

    function test() {
        const result = classifier.predict(X_test);
        const testSetLength = X_test.length
        const predictionError = error(result, Y_test);
        const accuracy = ((parseFloat(testSetLength)-parseFloat(predictionError))/parseFloat(testSetLength))*100;
	console.log(`Test Set Size = ${testSetLength} and accuracy ${accuracy}%`);
    }

    function error(predicted, expected) {
        let misclassifications = 0;
        for (var index = 0; index < predicted.length; index++) {
            console.log(`${index} => expected : ${expected[index]} and predicted : ${predicted[index]}`);
            if (predicted[index] != expected[index]) {
	        misclassifications++;
   	    }
	}
        return misclassifications;
    }

});
```

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

Source code 
===========

The whole source code is the following :
```
const fs = require("fs");
const hog = require("./hog/src");
const {default: Image} = require('image-js');
const SVM = require('libsvm-js/asm');

let options = {
    type: SVM.SVM_TYPES.NU_SVC, 
    kernel : SVM.KERNEL_TYPES.POLYNOMIAL,
    degree : 3,
    nu : 0.1,
    shrinking : false
}

let options_hog = {
    cellSize: 4,
    blockSize: 2,
    blockStride: 1,
    bins: 6,
    norm: "L2"
}

let X_train = [];
let Y_train = [];
let X_test = [];
let Y_test = [];

async function loadData(){
    // We will load the dataset

    async function loadTrainingSet(){
        var lines = fs.readFileSync('labels_train.csv').toString().split("\n");
        for(var i = 0; i < lines.length; i++){
            var elements = lines[i].split(";");
            if(elements.length < 2)
                continue;
            var file = __dirname + "/data/" + elements[0];
            // in the variable X, we will store the HOG of the pictures
            var image = await Image.load(file);
            image = await image.scale({width:100, height:100});
            var descriptor = hog.extractHOG(image, options_hog);
            X_train.push(descriptor);
            Y_train.push(elements[1]);
        }
    }

    async function loadTestSet(){
        var lines = fs.readFileSync('labels_test.csv').toString().split("\n");
        for(var i = 0; i < lines.length; i++){
            var elements = lines[i].split(";");
            if(elements.length < 2)
                continue;
            var file = __dirname + "/data/" + elements[0];
	    // in the variable X, we will store the HOG of the pictures
            var image = await Image.load(file);
            image = await image.scale({width:100, height:100});
            var descriptor = hog.extractHOG(image, options_hog);
            X_test.push(descriptor);
            Y_test.push(elements[1]);
        }
    }

    await loadTrainingSet();
    await loadTestSet();
}

loadData().then(function(){
    // Now, the dataset should be loaded. We will apply the classification
    // Begin of the classification

    var classifier = new SVM(options);

    classifier.train(X_train, Y_train);
    test();

    function test() {
        const result = classifier.predict(X_test);
        const testSetLength = X_test.length
        const predictionError = error(result, Y_test);
        const accuracy = ((parseFloat(testSetLength)-parseFloat(predictionError))/parseFloat(testSetLength))*100;
	console.log(`Test Set Size = ${testSetLength} and accuracy ${accuracy}%`);
    }

    function error(predicted, expected) {
        let misclassifications = 0;
        for (var index = 0; index < predicted.length; index++) {
            console.log(`${index} => expected : ${expected[index]} and predicted : ${predicted[index]}`);
            if (predicted[index] != expected[index]) {
	        misclassifications++;
   	    }
	}
        return misclassifications;
    }

});
```

Installation and running
=========================

**If you want to use the source code of this library, use the following instructions.** 
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
The library of the HOG descriptor will soon be available in the NPM. You can find the repository in the project [image-js](https://github.com/image-js/hog). The library used for the SVM is a part of the [mljs project](https://github.com/mljs). You can try other classifiers instead of SVM.
