# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:38:00 2018

@author: JGratsova
"""

# Import libraries
from sklearn import preprocessing
from pandas import DataFrame
import pandas as pd
import numpy as np
import pickle as pk
import time
#import sqlalchemy as sa
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint as sp_randInt
from scipy.stats import uniform as sp_randFloat
from pandas.plotting import scatter_matrix
import multiprocessing as mp

dataset = pd.read_csv("1to1_train_ready_rt7460.csv", header=None)
X = dataset.iloc[:354085, 1:805].values
y = dataset.iloc[:354085, 0].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4,
                                                    random_state = 0)
# show number of ones
y_test_count_1 = np.count_nonzero(y_test)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    KFold = 3
    model_list = []
    cv_outcomes = []
    description = []
    
###############################################################################
################  Manual tuning of parameter settings -PS1   ##################
###############################################################################
RF_1 = RandomForestClassifier(n_estimators = 10, criterion = 'gini',
                              max_depth = 6, min_samples_split = 2,
                              min_samples_leaf = 1,
                              min_weight_fraction_leaf = 0.0,
                              max_features = 'auto',max_leaf_nodes = None,
                              min_impurity_decrease = 0.0,
                              min_impurity_split = None, bootstrap = True,
                              oob_score = False, n_jobs = 10,
                              random_state = None, verbose = 1,
                              warm_start = False, class_weight = None)
model = RF_1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation
cv_results = cross_val_score(model, X_train, y_train,
                             cv = KFold, scoring = 'accuracy',
                             n_jobs = 10)
cv_outcomes.append(cv_results)
description.append('RF_1')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation Results
print("\n%s: " % ('Random Forest Algorithm: PS-1'))
prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
        cv_results.mean(), cv_results.std())

print(prt_string)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Train the model
trained_model = model.fit(X_train, y_train)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluate performance of the trained model
pred_class = trained_model.predict(X_test)
accuracy = accuracy_score(y_test, pred_class)
conf_matrix = confusion_matrix(y_test, pred_class)
class_report = classification_report(y_test, pred_class)
kappa_score = cohen_kappa_score(y_test, pred_class)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Collect performance results
model_list.append(('RF_1', 'Random Forest Algorithm: PS-1',
                  trained_model, accuracy, conf_matrix,
                  class_report, kappa_score))

###############################################################################
################  Manual tuning of parameter settings -PS2   ##################
###############################################################################

RF_2 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',
                              max_depth = 8, min_samples_split = 2,
                              min_samples_leaf = 1,
                              min_weight_fraction_leaf = 0.0,
                              max_features = None,max_leaf_nodes = None,
                              min_impurity_decrease = 0.0,
                              min_impurity_split = None, bootstrap = True,
                              oob_score = False, n_jobs = 10,
                              random_state = None, verbose = 1,
                              warm_start = False, class_weight = None)
model = RF_2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation
cv_results = cross_val_score(model, X_train, y_train,
                             cv = KFold, scoring = 'accuracy',
                             n_jobs = 10)
cv_outcomes.append(cv_results)
description.append('RF_2')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation Results
print("\n%s: " % ('Random Forest Algorithm: PS-2'))
prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
        cv_results.mean(), cv_results.std())

print(prt_string)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Train the model
trained_model = model.fit(X_train, y_train)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluate performance of the trained model
pred_class = trained_model.predict(X_test)
accuracy = accuracy_score(y_test, pred_class)
conf_matrix = confusion_matrix(y_test, pred_class)
class_report = classification_report(y_test, pred_class)
kappa_score = cohen_kappa_score(y_test, pred_class)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Collect performance results
model_list.append(('RF_2', 'Random Forest Algorithm: PS-2',
                  trained_model, accuracy, conf_matrix,
                  class_report, kappa_score))

###############################################################################
################  Manual tuning of parameter settings -PS3   ##################
###############################################################################

RF_3 = RandomForestClassifier(n_estimators = 1000, criterion = 'gini',
                              max_depth = 10, min_samples_split = 2,
                              min_samples_leaf = 1,
                              min_weight_fraction_leaf = 0.0,
                              max_features = 'log2',max_leaf_nodes = None,
                              min_impurity_decrease = 0.0,
                              min_impurity_split = None, bootstrap = True,
                              oob_score = False, n_jobs = 10,
                              random_state = None, verbose = 1,
                              warm_start = False, class_weight = None)
model = RF_3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation
cv_results = cross_val_score(model, X_train, y_train,
                             cv = KFold, scoring = 'accuracy',
                             n_jobs = 10)
cv_outcomes.append(cv_results)
description.append('RF_3')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation Results
print("\n%s: " % ('Random Forest Algorithm: PS-3'))
prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
        cv_results.mean(), cv_results.std())

print(prt_string)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Train the model
trained_model = model.fit(X_train, y_train)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluate performance of the trained model
pred_class = trained_model.predict(X_test)
accuracy = accuracy_score(y_test, pred_class)
conf_matrix = confusion_matrix(y_test, pred_class)
class_report = classification_report(y_test, pred_class)
kappa_score = cohen_kappa_score(y_test, pred_class)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Collect performance results
model_list.append(('RF_3', 'Random Forest Algorithm: PS-3',
                  trained_model, accuracy, conf_matrix,
                  class_report, kappa_score))

###############################################################################
#########  Automatic tuning of parameter settings using GridSearchCV   ########
###############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model setup
model = RandomForestClassifier()
parameters = {'max_depth' : [6, 10, 50],
              'criterion' : ['gini', 'entropy'],
              'max_features' : ['auto', 'sqrt', 'log2'],
              'n_estimators' : [100, 500, 1000]
              }

grid = GridSearchCV(estimator = model, param_grid = parameters,
                    cv = KFold, verbose = 1, n_jobs = 10)
grid.fit(X_train, y_train)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grid search results
print("\n =========================================================")
print(" Grid Search Results ")
print("============================================================")
print("\n The best estimator :\n",
      grid.best_estimator_)
print("\n The best score :\n",
      grid.best_score_)
print("\n The best parameters :\n",
      grid.best_params_)
print("\n =========================================================")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up model using grid search results
model = grid.best_estimator_

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross validation
cv_results = cross_val_score(model, X_train, y_train,
                             cv = KFold, scoring = 'accuracy',
                             verbose = 1, n_jobs = 10)
cv_outcomes.append(cv_results)
description.append('RF_4')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation Results
print("\n%s: " % ('Random Forest Algorithm: PS-4'))
prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
        cv_results.mean(), cv_results.std())

print(prt_string)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Train the model
trained_model = model.fit(X_train, y_train)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluate performance of the trained model
pred_class = trained_model.predict(X_test)
accuracy = accuracy_score(y_test, pred_class)
conf_matrix = confusion_matrix(y_test, pred_class)
class_report = classification_report(y_test, pred_class)
kappa_score = cohen_kappa_score(y_test, pred_class)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Collect performance results
model_list.append(('RF_4', 'Random Forest Algorithm: PS-4',
                  trained_model, accuracy, conf_matrix,
                  class_report, kappa_score))

###############################################################################
######  Automatic tuning of parameter settings using RandomizedSearchCV   #####
###############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model setup
model = RandomForestClassifier()
parameters = {'max_depth' : sp_randInt(4, 10),
              'criterion' : ['gini', 'entropy'],
              'max_features' : ['auto', 'sqrt', 'log2'],
              'n_estimators' : sp_randInt(100, 1000),
              'min_impurity_decrease' : sp_randFloat(),
              }


random = RandomizedSearchCV(estimator = model,
                            param_distributions = parameters,
                            cv = KFold,, n_iter = 10,
                            verbose = 1, n_jobs = 10)

random.fit(X_train, y_train)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Randomized search results
print("\n =========================================================")
print(" Random Search Results ")
print("============================================================")
print("\n The best estimator :\n",
      grid.best_estimator_)
print("\n The best score :\n",
      grid.best_score_)
print("\n The best parameters :\n",
      grid.best_params_)
print("\n =========================================================")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up model using randomized search results
model = random.best_estimator_

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross validation
cv_results = cross_val_score(model, X_train, y_train,
                             cv = KFold, scoring = 'accuracy',
                             verbose = 1, n_jobs = 10)
cv_outcomes.append(cv_results)
description.append('RF_5')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation Results
print("\n%s: " % ('RAndom Forest Algorithm: PS-5'))
prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
        cv_results.mean(), cv_results.std())

print(prt_string)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Train the model
trained_model = model.fit(X_train, y_train)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluate performance of the trained model
pred_class = trained_model.predict(X_test)
accuracy = accuracy_score(y_test, pred_class)
conf_matrix = confusion_matrix(y_test, pred_class)
class_report = classification_report(y_test, pred_class)
kappa_score = cohen_kappa_score(y_test, pred_class)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Collect performance results
model_list.append(('RF_5', 'Random Forest Algorithm: PS-5',
                  trained_model, accuracy, conf_matrix,
                  class_report, kappa_score))

###############################################################################
##############  Visualisation of results from Cross Validation   ##############
###############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the results
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison : Cross Validation Results')
ax = fig.add_subplot(111)
pyplot.boxplot(cv_outcomes, vert = False)
ax.set_yticklabels(description)
pyplot.show()

###############################################################################
#################  Trained Models : Evaluation and Reporting   ################
###############################################################################

print('\n Trained Models : Evaluation and Reporting ... ... ...')
for shtDes, des, model, accu, kappa, rept, cm in model_list:
    prt_ = "\nModel:{M}\nAccuracy:{A}\tKappa:{K}\nReport:\n{R}".format(
            M = des, A = round(accu, 2), K = round(kappa, 2), R = rept)
    prt_cm = "\nConfusion Matrix:\n{CM}".format(CM = cm)
    print(prt_, prt_cm)
    
    # Save the trained model
    with open('model_'+shtDes+'.pickle', 'wb') as f:
        pk.dump(model, f)
        
print("\n\nTrained models are saved ... Done ...")


            











