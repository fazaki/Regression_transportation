#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:08:07 2018

@author: fadi
"""
###############################################################################
## Libraries ##
###############
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.formula.api as sm
###############################################################################
## Reading Data ##
##################
os.chdir(r'D:\notebooks\ESRI-Taxi')
fnames = os.listdir("data")

dfs = []
for i in range(len(fnames)):
    dfs.append(pd.read_csv("data/"+fnames[i], usecols = [1,5,6,7], header = None, skip_blank_lines=True).dropna(axis=0, how='any').sample(frac = 0.005))

# Concatenate all data into one DataFrame
df1 = pd.concat(dfs, ignore_index=True)
df1.columns = ['datetime', 'PUloc', 'DOloc', 'Count']

###############################################################################
## Validation for NaN entries ##
################################
print "Validation for Null entries:"
print "PUtime empty count :", df1['datetime'].isnull().sum()
print "PUloc empty count :", df1['PUloc'].isnull().sum()
print "DOloc empty count :", df1['DOloc'].isnull().sum()
print "Count empty count :", df1['Count'].isnull().sum()

###############################################################################
## Explore data statistics ##
#############################
df1.describe()

print "Unique Pick-UP locations:",len(df1.PUloc.unique())
print "Unique Drop-off locations:",len(df1.DOloc.unique())

###############################################################################
## Handling Time ##
###################
df2 = df1.copy()
hours_range = 8

df2['datetime'] = pd.to_datetime(df1['datetime'])
df2['year'] = df2['datetime'].dt.year
df2['month'] = df2['datetime'].dt.month
df2['day'] = df2['datetime'].dt.day
df2['day_part'] = np.floor(df2['datetime'].dt.hour/hours_range)
df2['day_part'] = df2.day_part.astype(int)
df2['dayofweek'] = df2['datetime'].dt.dayofweek + 1 # Monday = 1, Sunday = 7
df2['week'] = df2['datetime'].dt.week
#let's describe the dat again
df2.describe()
###############################################################################
## Histogram Vizualization               ##   
## Inputs are:                           ##
## df --> dataframe name                 ##
## col --> column name                   ##
## n --> m for month, w for day of week  ##    
###########################################
def hst(df,col,n):
    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.countplot(x=col, data=df)
    if n == "w": 
        ax.axes.set_xticklabels(["MON", "TUE","WED","THU","FRI","SAT","SUN"])
    elif n == "m":
        ax.axes.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
###############################################################################
## Data Filtering on certain trajectory ##
##########################################

df3 = df2.drop(['datetime'],axis = 1)
df4 = df3.groupby(['PUloc','DOloc']).size().reset_index(name='Count').sort_values('Count', ascending=False).head(10)
df3 = df3[['PUloc','DOloc','year','month','day','day_part','dayofweek','week','Count']]
df5 = df3[(df3['PUloc'] == PUloc) & (df3['DOloc'] == DOloc)]

###############################################################################
## Machine Learning Part ##
###########################

def run_regression(name,input_dt):
    X = input_dt.iloc[:,[0,1,2,3,4,5,6,7]]
    Y = input_dt.iloc[:,[8]]
    Y = Y.values.reshape(len(X))
    validation_size = 0.20
    seed = 0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = validation_size, random_state = seed)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=0, loss='ls', warm_start = True)
    model.fit(X_train,Y_train)
    return name,model,model.score(X_test, Y_test)

# run model for all categories and put results into the table.
# also save trained models for later use

Name, model, R2 = run_regression('df3',df3)

###############################################################################