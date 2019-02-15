import numpy
import matplotlib.pyplot as pyplot
import cv2
import os
import random


#reads in training and test data and formats it into tensorflow readable numpys
def numpyBuilder(directory, xName, yName, labels):
    trainingData = []
    IMAGE_SIZE = 64
    for label in labels:
        path = os.path.join(directory, label)
        labelIndex = labels.index(label)
        for xray in os.listdir(path):
            try:
                #resizes images into 64x64 grayscale images
                imgArray = cv2.imread(os.path.join(path, xray), cv2.IMREAD_GRAYSCALE)
                resizedArray = cv2.resize(imgArray, (IMAGE_SIZE, IMAGE_SIZE))
                trainingData.append([resizedArray, labelIndex])
            except Exception as error:
                pass

    #shuffles data to prevent array order bias
    random.shuffle(trainingData)

    xNumpy = []
    yNumpy = []

    for imageArray, label in trainingData:
        xNumpy.append(imageArray)
        yNumpy.append(label)

    #adds channel index to numpy array
    xNumpy = numpy.array(xNumpy).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    #constricts values between 0 and 1
    xNumpy=xNumpy/255.0

    numpy.save(xName, xNumpy)
    numpy.save(yName, yNumpy)


labels = ["NORMAL", "PNEUMONIA"]
testDirectory = "F:\machineLearningData\chest_xray\\test"
trainingDirectory = "F:\machineLearningData\chest_xray\\train"
xTestingName = "xTest.npy"
yTestingName = "yTest.npy"
yTrainingName = "yTrain.npy"
xTrainingName = "xTrain.npy"

numpyBuilder(testDirectory, xTestingName, yTestingName, labels)
numpyBuilder(trainingDirectory, xTrainingName, yTrainingName, labels)
