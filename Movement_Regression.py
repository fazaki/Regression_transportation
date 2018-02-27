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
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
#import statsmodels.formula.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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
df3 = df3[['PUloc','DOloc','year','month','day','day_part','dayofweek','week','Count']]
df4 = df3.groupby(['PUloc','DOloc','year','month','day','day_part','dayofweek','week'])['Count'].sum().reset_index()
df5 = df4.drop(['year'],axis = 1)
df6 = df4.drop(['year','week'],axis = 1)


df10 = df4.groupby(['PUloc','DOloc']).size().reset_index(name='Count').sort_values('Count', ascending=False).head(10)

###############################################################################
## Machine Learning Part ##
###########################
## this function calculates the R2 for train and test

def run_regression_split(name,input_df):
    X = input_df.iloc[:,0:len(input_df.columns)-1]
    Y = input_df.iloc[:,-1]
    Y = Y.values.reshape(len(X))
    validation_size = 0.20
    seed = 0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = validation_size, random_state = seed)
    # Decision Tree model
    model = DecisionTreeRegressor()
    model.fit(X_train,Y_train)
    R2 = model.score(X_train,Y_train)
    R2_test = model.score(X_test,Y_test)
    print 'DataSet: {0}\tModel: {1}\t\tR2 train = {2:.3f}\tR2 test = {3:.3f}'.format(name, str(model).split('(')[0], R2, R2_test)

    for est in [10, 100, 200, 300]:
        
        ##########################
        ### Random forst Model ###
        ##########################
        model = RandomForestRegressor(n_estimators = est, random_state = 0)
        model.fit(X_train,Y_train)
        R2 = model.score(X_train,Y_train)
        R2_test = model.score(X_test,Y_test)
        print 'DataSet: {0}\tModel: {1}\t\t# of trees = {4}\tR2 train = {2:.3f}\tR2 test = {3:.3f}'.format(name, str(model).split('(')[0], R2, R2_test, est)

        ###############################
        ### Gradient Boosting Model ###
        ###############################
        model = GradientBoostingRegressor(n_estimators= est, learning_rate=0.01, random_state=0, loss='ls')
        model.fit(X_train,Y_train)
        R2 = model.score(X_train,Y_train)
        R2_test = model.score(X_test,Y_test)
        print 'DataSet: {0}\tModel: {1}\t# of trees = {4}\tR2 train = {2:.3f}\tR2 test = {3:.3f}'.format(name, str(model).split('(')[0], R2, R2_test, est)

###############################################################################
run_regression_split('df4',df4)
print "\n"
run_regression_split('df5',df5)
print "\n"
###############################################################################

## this function calculates the R2 for the whole data set

def run_regression(name,input_df):
    X = input_df.iloc[:,0:len(input_df.columns)-1]
    Y = input_df.iloc[:,-1]
    Y = Y.values.reshape(len(X))

    # Decision Tree model
    model = DecisionTreeRegressor()
    model.fit(X,Y)
    R2 = model.score(X,Y)
    print 'DataSet: {0}\tModel: {1}\t\tR2 = {2:.3f}'.format(name, str(model).split('(')[0], R2)

    for est in [10, 100, 200, 300]:
        
        ##########################
        ### Random forst Model ###
        ##########################
        model = RandomForestRegressor(n_estimators = est, random_state = 0)
        model.fit(X,Y)
        R2 = model.score(X,Y)
        print 'DataSet: {0}\tModel: {1}\t\tR2 = {2:.3f}\t# of trees = {3}'.format(name, str(model).split('(')[0], R2, est)
        
        ###############################
        ### Gradient Boosting Model ###
        ###############################
        model = GradientBoostingRegressor(n_estimators= est, learning_rate=0.01, random_state=0, loss='ls')
        model.fit(X,Y)
        R2 = model.score(X,Y)
        print 'DataSet: {0}\tModel: {1}\tR2 = {2:.3f}\t# of trees = {3}'.format(name, str(model).split('(')[0], R2, est)

###############################################################################

run_regression('df4',df4)
print "\n"
run_regression('df5',df5)
print "\n"

###############################################################################
### Random forst Model ###
##########################
X = df4.iloc[:,0:len(df4.columns)-1]
Y = df4.iloc[:,-1]
Y = Y.values.reshape(len(X))

model = RandomForestRegressor(n_estimators = 300, random_state = 0)
model.fit(X,Y)
R2 = model.score(X,Y)
print 'Model: {0}\t\tR2 = {1:.4f}'.format(str(model).split('(')[0], R2)

###############################################################################
### PREDICTION ###
##################

PUloc = 7
DOloc = 7
year = 2016
month = 7
day = 10
day_part = 0
dayofweek = 7
week = 27

input_ = [(PUloc, DOloc ,year ,month ,day ,day_part ,dayofweek, week)]

math.ceil(model.predict(input_)[0])

###############################################################################
###############################################################################