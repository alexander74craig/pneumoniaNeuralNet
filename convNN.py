import numpy
import tensorflow
import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from sklearn.utils import class_weight

#reads in training data
xTrain = numpy.load("xTrain.npy")
yTrain = numpy.load("yTrain.npy")

xTest = numpy.load("xTest.npy")
yTest = numpy.load("yTest.npy")

#computes class weights to prevent bias towards the larger training set.
class_weights = class_weight.compute_class_weight('balanced', numpy.unique(yTrain), yTrain)
d_class_weights = dict(enumerate(class_weights))

#builds model
model = Sequential()

model.add(Conv2D(16, (3,3), activation = "relu", input_shape = xTrain.shape[1:]))
#each layer has aggressive dropout to prevent over-fitting
model.add(Dropout(rate = 0.5))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), activation = "relu"))
model.add(Dropout(rate = 0.5))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation = "relu"))
model.add(Dropout(rate = 0.5))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(rate = 0.5))

model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer = "adam",
            loss = 'binary_crossentropy',
            metrics = ['accuracy'])

#trains model and validates training cycles with test data
model.fit(xTrain, yTrain, epochs = 20, class_weight=d_class_weights, batch_size = 64, validation_data = (xTest, yTest))

modelName = "pneumonia{0}.model".format(time.strftime("%Y%m%d-%H%M%S"))
model.save(modelName)