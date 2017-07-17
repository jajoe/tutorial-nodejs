Introduction to classification with mljs
========================================

This tutorial will solve a problem of Machine Learning called classification. Classification is the problem of identifying to which of a set of categories a new observation belongs (Wikipedia).

We want to solve a classification with JS. The main advantage of using JS is that it enables to solve a problem in a browser, so it can be done client-side.

We will use functions from ML.JS (https://github.com/mljs), particularly libsvm-js, ml-naivebayes and ml-knn.

Dataset
=======

We will try to define the type of leafs thanks to their features.

Dataset used : https://archive.ics.uci.edu/ml/datasets/Leaf


In this dataset, we have 340 data, with 16 features each. The first feature is the class (i.e the type of the leaf). The goal of the classification is to determine the class of a given leaf thanks to the 15 others features of this leaf.

On the previous link, you can find the description of each feature. These features are float or number.

Note :
One of the difficulties of this dataset is that we haven't a lot of data.


Installation :
==============

```
npm install libsvm-js ml-naivebayes ml-knn csvtojson 
```

First step :
============

The first step will be to load the dataset and to delete the useless feature. Indeed, the second feature is useless, because it's just created to identify a data. We will use the package csvtojson to import and manipulate the data.

The data are in a csv file without header. Here is how to import it :

```
const csv = require('csvtojson');

const csvFilePath = 'leaf.csv'; // Data
const names = ['type', 'specimenNumber', 'eccentricity', 'aspectRatio', 'elongation', 'solidity', 'stochasticConvexity', 'isoperimetricFactor', 'maxIndetationDepth', 'lobedness', 'intensity', 'contrast', 'smoothness', 'thirdMoment', 'uniformity', 'entropy']; // For header

csv({noheader: true, headers: names})
    .fromFile(csvFilePath)
    .on('json', (jsonObj) => {
        data.push(jsonObj); // Push each object to data Array
    })
    .on('done', (error) => {
        seperationSize = 0.9 * data.length;
        data = shuffleArray(data);
        dressData();
    });

function dressData() {
    let types = new Set(); // To gather UNIQUE classes
    data.forEach((row) => {
        types.add(row.type);
    });
    typesArray = [...types]; // To save the different types of classes.

    data.forEach((row) => {
        let rowArray, typeNumber;
        rowArray = Object.keys(row).map(key => parseFloat(row[key])).slice(2, 16); // We don't use the 2 first elements, which are the type (i.e class) and the specimen number (i.e ID)
        typeNumber = typesArray.indexOf(row.type); // Convert type(String) to type(Number)

        X.push(rowArray);
        y.push(typeNumber);
    });

    trainingSetX = X.slice(0, seperationSize);
    trainingSetY = y.slice(0, seperationSize);
    testSetX = X.slice(seperationSize);
    testSetY = y.slice(seperationSize);

}
``` 

We use the function shuffleArray to shuffle the dataset to allow splitting. If you run many times the script, results will be different because the separation between training set and test set will be different.

Second step - Configuration of the model :
==========================================

We will try three different methods : the SVM classifier and the naive bayes classifier.

SVM
---

```
const SVM = require('libsvm-js/asm');

let options = {
    kernel : SVM.KERNEL_TYPES.POLYNOMIAL,
    degree : 3,
    gamma : 20,
    cost : 100,
    shrinking : false
}

svm = new SVM(options);
```


Bayes
-----
```
const Bayes = require('ml-naivebayes');

bayes = new Bayes();
```

Note : We don't need to give options to the naive bayes classifier.

KNN
---
```
knn = new KNN(trainingSetX, trainingSetY, {k:5});
```

Note : We need to give the training set when we instanciate a KNN classifier.

Training and Evaluation :
=========================

We have our model. Now we can train it with the training data :

SVM
---
```
svm.train(trainingSetX, trainingSetY);
```

Bayes
-----
```
bayes.train(trainingSetX, trainingSetY);
```

KNN
-----
The model is trained after the instanciation of the KNN-classifier.

Evaluation
-----------

When the model is trained, we can use it on the test set. We will predict the labels of these data and compare the predicted labels with the expected labels. Here is how we do that :

```
function test() {
    const result = svm.predict(testSetX);
    const testSetLength = testSetX.length
    const predictionError = error(result, testSetY);
    console.log(`Test Set Size = ${testSetLength} and number of Misclassifications = ${predictionError}`);
}

function error(predicted, expected) {
    let misclassifications = 0;
    for (var index = 0; index < predicted.length; index++) {
        console.log(`truth : ${expected[index]} and prediction : ${predicted[index]}`);
        if (predicted[index] !== expected[index]) {
            misclassifications++;
        }
    }
    return misclassifications;
}
```
The script will display for each data from the test set the expected labels and the predicted labels.

Results :
=========

The accuracy is the percentage of right predictions. We have the number of test data (let's call it N) and the number of misclassification (let's call it f), so the accucary is (f/N)*100 %.
 
SVM
---
There are on average 7 or 8 errors on 34 predictions, so the accuracy is ~78%. (the result oscillate between 4 and 14, it depends of the splitting into the training and test sets)

Bayes
-----
The results are very bad, only on average between 26% and 41% of right predictions. This method give bad results with this dataset. The naive baye works generally fine with text classification.

KNN 
---
With KNN (k=5), the result is generally between 47% and 59%, and that's not very good.

Conclusion :
============

This tutorial show an example of solving of a classification problem with SVM and naive bayes classifier. You can use a lot of others Machine Learning methods to solve this problem (random forests, KNN, neural networks...) and the parameters given to this SVM in my example are probably not the best, but the goal of this tutorial is only to let you see how to use ML.JS to solve classification problems (you can have better precision than 78%).


Note :
======

Thanks to Abhishek Soni for the two tutorials https://hackernoon.com/machine-learning-with-javascript-part-1-9b97f3ed4fe5 and https://hackernoon.com/machine-learning-with-javascript-part-2-da994c17d483. My code reuse parts of the second tutorial (which is the solving of a classification problem with the method of KNN).
