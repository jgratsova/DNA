
#y1 = np.random.randint(1, size=(44044, 1))

import pandas as pd
import numpy
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame

#dataset = pd.DataFrame(sample_vector_5_7_15)
dataset = pd.read_csv("1to1_train_ready_rt7460.csv", header=None)
X = dataset.iloc[:354085, 1:805].values
y = dataset.iloc[:354085, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
# show number of ones
y_test_count_1 = numpy.count_nonzero(y_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.utils import class_weight

#class_weight = {0: 1.,
#
#                1: 60.}
                
class_weight = class_weight.compute_class_weight('balanced',numpy.unique(y_train), y_train)

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer (with dropout)
classifier.add(Dense(200, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 804))
classifier.add(Dense(200, kernel_initializer = 'uniform', activation = 'sigmoid'))
#classifier.add(Dropout(rate = 0.05))
# Adding the output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#classifier.add(Dropout(rate = 0.05))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#classifier.add(Dropout(rate = 0.1))
# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, class_weight=class_weight, batch_size = 10, epochs = 50, verbose = 1)
#score = classifier.evaluate(X_test, y_test, batch_size=32)

# Plot results
import matplotlib.pyplot as plt

def plot_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(30, 16))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['loss'][-1]
    acc = history.history['acc'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))
    
# plot accuracy loss  
plot_loss_accuracy(history)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.6)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_score
cm_ANN_200_ = confusion_matrix(y_test, y_pred)
sl_ANN_200_ = brier_score_loss(y_test, y_pred)
ps_ANN_200_ = precision_score(y_test, y_pred)
























