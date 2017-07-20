const SVM = require('libsvm-js/asm');
const csv = require('csvtojson');
const PCA = require('ml-pca');

let svm;
let options = {
    kernel : SVM.KERNEL_TYPES.RBF,
    gamma : 1e-2,
    cost : 1,
    shrinking : true
}
let pca;

const csvFilePath = 'data.csv'; // Data
const names = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "label"]; // For header

let seperationSize;
let data = [], X = [], y = [];
let trainingSetX = [], trainingSetY = [], testSetX = [], testSetY = [];

csv({noheader: true, headers: names})
    .fromFile(csvFilePath)
    .on('json', (jsonObj) => {
        data.push(jsonObj); // Push each object to data Array
    })
    .on('done', (error) => {
        seperationSize = 0.2 * data.length;
        data = shuffleArray(data);
        dressData();
    });

function dressData() {
    let types = new Set(); // To gather UNIQUE classes
    data.forEach((row) => {
        types.add(row.label);
    });
    typesArray = [...types]; // To save the different types of classes.

    data.forEach((row) => {
        let rowArray, typeNumber;
        rowArray = Object.keys(row).map(key => parseFloat(row[key])).slice(1, 24); // We don't use the 1 first element1, which are the the ID
        typeNumber = typesArray.indexOf(row.label); // Convert type(String) to type(Number)

        X.push(rowArray);
        y.push(typeNumber);
    });

    trainingSetX = X.slice(0, seperationSize);
    trainingSetY = y.slice(0, seperationSize);
    testSetX = X.slice(seperationSize);
    testSetY = y.slice(seperationSize);

    train();
}

function train() {
    pca = new PCA(trainingSetX);
    trainingSetX = pca.predict(trainingSetX).subMatrixColumn([0, 1]); // PCA with 2 components
    svm = new SVM(options);
    svm.train(trainingSetX, trainingSetY);
    test();
}

function test() {
    testSetX = pca.predict(testSetX).subMatrixColumn([0, 1]); // PCA with 2 components
    const result = svm.predict(testSetX);
    const testSetLength = testSetX.length;
    const predictionError = error(result, testSetY);
    console.log(`Test Set Size = ${testSetLength} and number of Misclassifications = ${predictionError}`);
    console.log((parseFloat(testSetLength - predictionError)/parseFloat(testSetLength)).toString()+" % accuracy");
}

function error(predicted, expected) {
    let misclassifications = 0;
    for (var index = 0; index < predicted.length; index++) {
        //console.log(`truth : ${expected[index]} and prediction : ${predicted[index]}`);
        if (predicted[index] !== expected[index]) {
            misclassifications++;
        }
    }
    return misclassifications;
}

function shuffleArray(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
}
