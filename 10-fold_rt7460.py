


import pandas as pd
import numpy
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame

# --------------------------- 10-fold classifier ------------------------#

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
from sklearn.utils import class_weight


import pandas as pd
import numpy
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame

#dataset = pd.DataFrame(sample_vector_5_7_15)
df1 = pd.read_csv("1to1_train_ready_rt7460.csv", header = None)
df1.tail(5)
df2.head(5)
df2 = pd.read_csv("1to1_train_ready_rt11210.csv", header = None)
frames = [df1, df2]
dataset = pd.concat(frames)
X = dataset.iloc[:546301, 1:805].values
y = dataset.iloc[:546301, 0].values

X_train = dataset.iloc[:354084, 1:805].values
X_test = dataset.iloc[354084:, 1:805].values
y_train = dataset.iloc[:354084, 0].values
y_test = dataset.iloc[354084:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
# show number of ones
y_test_count_1 = numpy.count_nonzero(y_test)
#class_weight = {0: 1.,
#
#                1: 60.}
                
class_weight = class_weight.compute_class_weight('balanced',numpy.unique(y_train), y_train)

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# define 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X_train, y_train):
  # create model
	model = Sequential()
	model.add(Dense(200, input_dim=804, activation='sigmoid'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X_train, y_train, class_weight=class_weight, epochs=50, batch_size=10, verbose=1)
	# evaluate the model
	scores = model.evaluate(X_train, y_train, verbose=1)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.6)

# Making the Confusion Matrix and calculating accuracy/loss scores
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import scikitplot as skplt

cm_kfold = confusion_matrix(y_test, y_pred)
sl_kfold = brier_score_loss(y_test, y_pred)
ps_kfold = precision_score(y_test, y_pred)


# ---------------------------- Plotting ROC curves to compare K-fold and RF ------------------#

from sklearn.metrics import roc_curve
y_pred_kfold = model.predict(X_test).ravel()
fpr_kfold, tpr_kfold, thresholds_kfold = roc_curve(y_test, y_pred_kfold)

from sklearn.metrics import auc
auc_kfold = auc(fpr_kfold, tpr_kfold)

# plot ROC curves to compare K-fold and RF
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_kfold, tpr_kfold, label='K-Fold (area = {:.3f})'.format(auc_kfold))
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# Zoom in view of the upper left corner of the ROC curves comparison
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_kfold, tpr_kfold, label='K-Fold (area = {:.3f})'.format(auc_kfold))
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
























