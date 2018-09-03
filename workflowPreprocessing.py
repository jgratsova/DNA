# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:30:38 2018

@author: JGratsova
"""

# Import the libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
import random

###############################################################################
############  Dataset normalization and splitting into Train/Test   ###########
###############################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read file into pandas dataframe
df = pd.read_csv("1_to1_sample_rt7460.csv", header=None)

# Split dataframes by name and data points, removing first 2600 data points
pointsDf = df[df.columns[2602:]]
namesDf = df[df.columns[0:2]]

# Check that dataframe type is int
pointsDf.dtypes

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Apply normalization
X_train = np.array(pointsDf).transpose()
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train).transpose()

# Convert object X_train_minmax into a dataframe
df2= DataFrame(X_train_minmax, dtype='float')

# Concatenate normalised numerical data and names dataframe back together
normDf = pd.concat([namesDf, df2], axis=1)
normDf.dtypes

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Group samples by splitting the dataframe into a list of dataframes
splitDf = np.array_split(normDf, int(len(normDf)/4))

# Fraction of sample to retain for unseen dataset
sizeUnseen = int(len(splitDf)*0.2)

# Create unseen and train datasets
unseenData = []
trainData = []
unseenList = sorted(random.sample(range(len(splitDf)), sizeUnseen))
for i in range(len(splitDf)):
    if (i in unseenList):
        unseenData.append(splitDf[i])
    else:
        trainData.append(splitDf[i])

# Concatenate results back into dataframes and export into .csv files
unseen_Data = pd.concat(unseenData)
train_Data = pd.concat(trainData)
unseen_Data.to_csv('unseen_data_rt7460_1to1_2contributors.csv', index = False)
train_Data.to_csv('train_data_rt7460_1to1_2contributors.csv', index = False)


###############################################################################
#########  Feature selection and binning using sliding window method   ########
###############################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Stacking function
def featureStack(calls, sample_name,chan1,chan2,chan3,chan4):
    """
    Take calls data, extract required sample and
    combine channel data with peaks data
    """
    
    grouped = calls.groupby(['Sample','Channel'])
    channelPeaks = []
    for i in range(1,5):
        try:
            sample=grouped.get_group((sample_name, i))
        except KeyError as err:
            # catch error where a sample doesn't have peak calls
            # due to missing run data from OSIRIS (TO FIX)
            return(-1)
        
        peaks = np.round_(sample['Time'].values,0)
        peaks = peaks.astype('int')-2700
        pminus1 = peaks-1
        pplus1 = peaks+1
        peaks = list(pminus1) + list(peaks) + list(pplus1)
        peaksVector = np.random.randint(1, size = (4659, 1))
        peaksVector[peaks] = 1
        channelPeaks.append(peaksVector)
    
    featureVec1 = np.hstack((channelPeaks[0],chan1,chan2,chan3,chan4))
    featureVec2 = np.hstack((channelPeaks[1],chan2,chan1,chan3,chan4))
    featureVec3 = np.hstack((channelPeaks[2],chan3,chan1,chan2,chan4))
    featureVec4 = np.hstack((channelPeaks[3],chan4,chan1,chan2,chan3))
    
    sample_vector = np.vstack((featureVec1, featureVec2, featureVec3, featureVec4))
    return(sample_vector)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
# Read allele calls data into pandas dataframe
calls = pd.read_csv('allele_calls_rt7460_1to1_2contributors.csv', sep='\,')

# Read normalized dataset file into pandas dataframe
train_df = pd.read_csv("train_data_rt7460_1to1_2contributors.csv")

# Create an indexer frame for further sliding window splitting
indexer = np.arange(201)[None, :] + 1*np.arange(4659)[:, None]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Split data and re-stack it to fit the indexer frame
d = []
c = []
i = 0
chunk = 0
all_samples_vec = np.zeros(2)

for idx, line in train_df.iterrows():
    sample_name = line[0]
    sample_name = sample_name[:-4] + ".fsa"
    data = line[2:4863].values
    b = np.array(data.flatten())    
    c.append(b[indexer])
    i = i + 1
    if (i == 4):
        print(sample_name)
        temp = np.hstack((c))
        if (len(d)):
            d = np.vstack((d,temp))
        else:
            d = temp
        i = 0
        c = []
        
        # Group data by channel
        chan1 = d[:, 0:201]
        chan2 = d[:, 201:402]
        chan3 = d[:, 402:603]
        chan4 = d[:, 603:804]
        sample_vec = featureStack(calls, sample_name, chan1, chan2, chan3, chan4)
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        # Check for errors
        if (type(sample_vec) is not np.ndarray):
            if (sample_vec == -1):
                # If sample doesn't exist in allele calls skip for time being
                d = []
                continue
            else:
                raise ValueError('ERROR - sample_vec is not as expected')
                exit(1)
        
        # Check if first sample or not
        if (len(all_samples_vec) != 2):
            # If not the first sample, append 
            all_samples_vec = np.vstack((all_samples_vec, sample_vec))
        else:
            # First sample
            all_samples_vec = sample_vec
            
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~            
        # Output the results into a CSV file, using chunking method
        d = []
        chunk = chunk + 1
        if (chunk % 10 == 0):
            # for every 10th sample write to file and reset all_samples_vec
            # to save memory errors
            print("Outputting chunk...")
            df3 = DataFrame(all_samples_vec)
            df3.to_csv('train_ready_rt7460_1to1_2contributors.csv', 
                       index = False, mode='a', header = False)
            del(df3)
            all_samples_vec = np.zeros(2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
# Convert to pandas dataframe and export into a CSV file format
if (len(all_samples_vec) != 2):
    print("Outputting remainder...")
    final_df = pd.DataFrame(all_samples_vec)
    final_df.head(10)
    final_df.to_csv('train_ready__1to1_2contributors.csv', 
                    index = False, mode='a', header = False)
