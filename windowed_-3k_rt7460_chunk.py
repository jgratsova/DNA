#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:25:44 2018

@author: julia
"""

# Importing the libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
 #import more_itertools as mit
#from itertools import chain
from numpy import split

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


# read allele calls data into a dataframe
calls = pd.read_csv('allele_calls_rt7460.csv', sep='\,')

# read file into pandas dataframe
train_df = pd.read_csv("train_data_rt7460_test.csv")

# create an indexer frame for further sliding window splitting
indexer = np.arange(201)[None, :] + 1*np.arange(4659)[:, None]

# split data and re-stack it to fit the indexer frame
d = []
i = 0
c = []
chunk = 0
all_samples_vec = np.zeros(2)
for idx, line in train_df.iterrows():
    sample_name = line[0]
    sample_name = sample_name[:-4] + ".fsa"
    data = line[2:4863].values
    b = np.array(data.flatten())    
#for line, in (c):
    c.append(b[indexer])
    i = i + 1
    #print(i)
    if (i == 4):
        print(sample_name)
        temp = np.hstack((c))
        if (len(d)):
            d = np.vstack((d,temp))
        else:
            d = temp
        i = 0
        c = []
        
        # group data by channel
        chan1 = d[:, 0:201]
        chan2 = d[:, 201:402]
        chan3 = d[:, 402:603]
        chan4 = d[:, 603:804]
        sample_vec = featureStack(calls, sample_name, chan1, chan2, chan3, chan4)
        #print(type(sample_vec))
        if (type(sample_vec) is not np.ndarray):
            if (sample_vec == -1):
                # sample doesn't exist in allele calls skip for time being
                d = []
                continue
            else:
                raise ValueError('ERROR - sample_vec is not as expected')
                exit(1)
        
        # check if first sample or not
        if (len(all_samples_vec) != 2):
            # not first sample, append 
            all_samples_vec = np.vstack((all_samples_vec, sample_vec))
        else:
            # first sample
            all_samples_vec = sample_vec
        
        d = []
        chunk = chunk + 1
        #print(chunk)
        if (chunk % 30 == 0):
            # for every 10th sample write to file and reset all_samples_vec
            # to save memory errors
            print("Outputting chunk...")
            df = DataFrame(all_samples_vec)
            df.to_csv('train_ready_rt7460.csv', index = False, mode='a', header = False)
            del(df)
            all_samples_vec = np.zeros(2)
        #break
    
# convert to dataframe and export into a .csv file format
if (len(all_samples_vec) != 2):
    print("Outputting remainder...")
    final_df = pd.DataFrame(all_samples_vec)
    final_df.head(10)
    final_df.to_csv('train_ready_rt7460_test.csv', index = False, mode='a', header = False)
