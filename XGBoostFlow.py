# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:34:00 2018

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
from scipy.stats import randint as sp_randInt
from scipy.stats import uniform as sp_randFloat
from pandas.plotting import scatter_matrix
import xgboost as xgb

dataset = pd.read_csv("1to1_train_ready_rt7460.csv", header=None)
X = dataset.iloc[:354085, 1:805].values
y = dataset.iloc[:354085, 0].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
# show number of ones
y_test_count_1 = np.count_nonzero(y_test)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    KFold = 4
    model_list = []
    cv_outcomes = []
    description = []
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Boosting Algorithms - XGBoost

###############################################################################
################  Manual tuning of parameter settings -PS1   ##################
###############################################################################

xgmat = xgb.DMatrix(X_train, label = y_train)
param = {}
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'logloss'
param['silent'] = 1
param['nthread'] = 4 
param['eta'] = 0.01
param['gamma'] = 0.5
param['max_depth'] = 6
param['min_child_weight'] = 0.2
param['max_delta_step'] = 0.1
param['subsample'] = 0.8
param['colsample_bytree'] = 0.9
param['lambda'] = 0.8
param['alpha'] = 0.5

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation
cv_results = xgb.cv(params = param, dtrain = xgmat,
                    num_boost_round = 8000, nfold = KFold,
                    metrics = ['logloss'], early_stopping_rounds = 20,
                    verbose_eval = False)

cv_outcomes.append(1- cv_results.mean())
description.append('XGBoost_1')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation Results
print("\n%s: " % ('XGBoost Algorithm: PS-1'))
print("CV Mean Error Rate: %f (Std: %f)"% (
        cv_results.mean()))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Train the model
trained_model = xgb.train(param, xgmat, cv_results.shape[0])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluate the trained model results
xgmat = xgb.DMatrix(X_test)
pred_class_prob = trained_model.predict(xgmat)
threshold = 0.5
pred_class = (pd.Series(pred_class_prob) > threshold).map({True: 1, False: 0})


accuracy = accuracy_score(y_test, pred_class)
conf_matrix = confusion_matrix(y_test, pred_class)
class_report = classification_report(y_test, pred_class)
kappa_score = cohen_kappa_score(y_test, pred_class)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Collect performance results
model_list.append(('XGBoost_1', 'XGBoost Algorithm: PS-1',
                  trained_model, accuracy, conf_matrix,
                  class_report, kappa_score))

###############################################################################
################  Manual tuning of parameter settings -PS2   ##################
###############################################################################

xgmat = xgb.DMatrix(X_train, label = y_train)
param = {}
param['objective'] = 'multi:softmax'
param['eval_metric'] = 'merror'
param['num_class'] = 2
param['silent'] = 1
param['nthread'] = 4 
param['eta'] = 0.01
param['gamma'] = 0.5
param['max_depth'] = 8
param['min_child_weight'] = 0.2
param['max_delta_step'] = 0.1
param['subsample'] = 0.8
param['colsample_bytree'] = 0.9
param['lambda'] = 0.8
param['alpha'] = 0.5

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation
cv_results = xgb.cv(params = param, dtrain = xgmat,
                    num_boost_round = 8000, nfold = KFold,
                    metrics = ['merror'], early_stopping_rounds = 20,
                    verbose_eval = False)

cv_outcomes.append(1- cv_results.mean())
description.append('XGBoost_2')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation Results
print("\n%s: " % ('XGBoost Algorithm: PS-2'))
print("CV Mean Error Rate: %f (Std: %f)"% (
        cv_results.mean()))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Train the model
trained_model = xgb.train(param, xgmat, cv_results.shape[0])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluate the trained model results
xgmat = xgb.DMatrix(X_test)
pred_class = trained_model.predict(xgmat)
accuracy = accuracy_score(y_test, pred_class)
conf_matrix = confusion_matrix(y_test, pred_class)
class_report = classification_report(y_test, pred_class)
kappa_score = cohen_kappa_score(y_test, pred_class)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Collect performance results
model_list.append(('XGBoost_2', 'XGBoost Algorithm: PS-2',
                  trained_model, accuracy, conf_matrix,
                  class_report, kappa_score))

###############################################################################
################  Manual tuning of parameter settings -PS1   ##################
###############################################################################

XGBoost_3 = xgb.XGBClassifier(gamma = 0.5, learning_rate = 0.15,
                              max_delta_step = 0, max_depth = 6,
                              min_child_weight = 1, n_estimators = 800,
                              nthread = 1, objective = 'binary:logistic',
                              reg_alpha = 0, reg_lambda = 1,
                              scale_pos_weight = 1, silent = True,
                              subsample = 1)
model = XGBoost_3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation
cv_results = cross_val_score(model, X_train, y_train,
                             cv = KFold, scoring = 'accuracy',
                             n_jobs = 10)

cv_outcomes.append(cv_results)
description.append('XGBoost_3')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation Results
print("\n%s: " % ('XGBoost Algorithm: PS-3'))
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
model_list.append(('XGBoost_3', 'XGBoost Algorithm: PS-3',
                  trained_model, accuracy, conf_matrix,
                  class_report, kappa_score))

###############################################################################
########  Automatic tuning of parameter settings using GridSearchCV   #########
###############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model setup
model = xgb.XGBClassifier()
parameters = {'max_depth' : [6, 8, 10],
              'gamma' :[0.2, 0.5, 0.6],
              'learning_rate' : [0.01, 0.05, 0.1],
              'n_estimators' : [100, 500, 1000]
              }

grid = GridSearchCV(estimator = model, param_grid = parameters, cv = KFold)
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
description.append('XGBoost_4')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation Results
print("\n%s: " % ('XGBoost Algorithm: PS-4'))
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
model_list.append(('XGBoost_4', 'XGBoost Algorithm: PS-4',
                  trained_model, accuracy, conf_matrix,
                  class_report, kappa_score))

###############################################################################
######  Automatic tuning of parameter settings using RandomizedSearchCV   #####
###############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model setup
model = xgb.XGBClassifier()
parameters = {'max_depth' : sp_randInt(4, 10),
              'gamma' :sp_randFloat(),
              'learning_rate' : sp_randFloat(),
              'n_estimators' : sp_randInt(100, 1000)
              }

random = RandomizedSearchCV(estimator = model,
                            param_distributions = parameters,
                            cv = KFold, verbose = 1, n_iter = 10)

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
description.append('XGBoost_5')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross Validation Results
print("\n%s: " % ('XGBoost Algorithm: PS-5'))
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
model_list.append(('XGBoost_5', 'XGBoost Algorithm: PS-5',
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


 

