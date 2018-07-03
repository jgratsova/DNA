# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 10:03:55 2018

@author: JGratsova
"""

 # --------------------- Pickle single object -----------#
 
import pickle

with open('filename.pickle', 'wb') as f:
    # Pickle an object using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    
with open('1to1_7460_df.pickle', 'rb') as f:
    # The protocol version used is detected automatically
    data = pickle.load(f)