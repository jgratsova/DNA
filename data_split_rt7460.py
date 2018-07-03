# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:02:21 2018

@author: JGratsova
"""

# Importing the libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
import random

# read file into pandas dataframe
df = pd.read_csv("1_to1_samples_rt7460.csv", header=None)

# split dataframes by name and data points, removing first 3000 data points
pointsdf = df[df.columns[2602:]]
namesdf = df[df.columns[0:2]]

#check that df type is int
pointsdf.dtypes

# apply normalization
X_train = np.array(pointsdf).transpose()
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train).transpose()

# convert object X_train_minmax into a dataframe
df2= DataFrame(X_train_minmax, dtype='float')

# concatenate normalised numerical data and names dataframe back together
normDF = pd.concat([namesdf, df2], axis=1)
normDF.dtypes

# group samples by splitting the dataframe into a list of dataframes
split_df = np.array_split(normDF, int(len(normDF)/4))

# fraction of sample to retain for unseen dataset
sizeUnseen = int(len(split_df)*0.2)

# create unseen and train datasets
unseenData = []
trainData = []
unseenList = sorted(random.sample(range(len(split_df)), sizeUnseen))
for i in range(len(split_df)):
    if (i in unseenList):
        unseenData.append(split_df[i])
    else:
        trainData.append(split_df[i])

# concatenate results back into dataframes and export into .csv files
unseen_Data = pd.concat(unseenData)
train_Data = pd.concat(trainData)
unseen_Data.to_csv('unseen_data_rt7460_test.csv', index = False)
train_Data.to_csv('train_data_rt7460_test.csv', index = False)




































    