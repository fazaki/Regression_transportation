#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:08:07 2018

@author: fadi
"""
###############################################################################
## Libraries ##

import pandas as pd
import random
import matplotlib.pyplot as plt

###############################################################################
## Reading Data ##

filename = "green_tripdata_2017-06.csv"

df = pd.read_csv(filename , \
                 usecols = [1,2,5,6,7], \
                 header = 0, \
                 skip_blank_lines=True)\
                 .dropna(axis=0, how='any')\
                 .sample(frac = 0.001)

###############################################################################                 
## Remanming columns ##

df.columns = ['PUtime','DOtime', 'PUloc', 'DOloc', 'Count']             
    
###############################################################################
## Validation for NaN entries ##

print "Validation for Null entries:"
print "PUtime empty count :", df['PUtime'].isnull().sum()
print "DOtime empty count :", df['DOtime'].isnull().sum()
print "PUloc empty count :", df['PUloc'].isnull().sum()
print "DOloc empty count :", df['DOloc'].isnull().sum()
print "Count empty count :", df['Count'].isnull().sum()

###############################################################################
## Explore data statistics ##

df.describe()
unique_puloc = pd.DataFrame(df.PUloc.unique()).sort([0], ascending =  True)
unique_doloc = pd.DataFrame(df.DOloc.unique()).sort([0], ascending =  True)
print "Unique Pick-UP locations:",len(unique_puloc)
print "Unique Drop-off locations:",len(unique_doloc)

###############################################################################
