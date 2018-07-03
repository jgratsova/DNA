# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:57:49 2018

@author: JGratsova
"""
# ----------------- Random Forest classifier --------------------- #

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import pandas as pd
import numpy
from pandas import DataFrame
from sklearn.utils import class_weight

#dataset = pd.DataFrame(sample_vector_5_7_15)
dataset = pd.read_csv("1to1_train_ready_rt7460.csv", header=None)
X = dataset.iloc[:354085, 1:805].values
y = dataset.iloc[:354085, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
# show number of ones
y_test_count_1 = numpy.count_nonzero(y_test)
# build the classifier
rf_model = RandomForestClassifier(max_depth=55, n_estimators=500, verbose =1, random_state = 0, n_jobs = 10)
rf_model.fit(X_train, y_train)
print ("rf_model :: "), rf_model

# perform predictions
rf_predictions = rf_model.predict(X_test)
for i in range (len(rf_predictions)):
    #Find false positives
    if (list(y_test)[i] == 0 and rf_predictions[i] == 1):
        print("FP at {}".format(i))
    print ("Actual outcome :: {} and Predicted outcome :: {} ". format(list(y_test)[i], rf_predictions[i]))
    
# calculate accuracy and loss
print("Train Accuracy :: ", accuracy_score(y_train, rf_model.predict(X_train)))
print("Test Accuracy :: ", accuracy_score(y_test, rf_predictions))
print("Test loss :: ", brier_score_loss(y_test, rf_predictions))

# make confusion matrix
print("Confusion matrix ", confusion_matrix(y_test, rf_predictions))
cm_rf_md_55 = confusion_matrix(y_test, rf_predictions)

# calculate ROC
y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
auc_rf = auc(fpr_rf, tpr_rf)


# ------------------------------------ Random Hyperparameter Grid-----------------------#
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint

# create a parameter grid to sample from during fitting
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
n_estimators = 500
# Number of features to consider at every split
#max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
#min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
              # 'max_features': max_features,
               'max_depth': max_depth,
              # 'min_samples_split': min_samples_split,
              # 'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
pprint(random_grid)

# instantiate the random search and fit
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 10)
# Fit the random search model
rf_random.fit(X_train, y_train)

#view the best parameters from fitting the random search
rf_random.best_params_