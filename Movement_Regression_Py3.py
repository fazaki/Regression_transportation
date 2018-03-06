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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import random
import arcpy
import io
###############################################################################
## Reading Data ##
##################
os.chdir(r'D:\Notebooks\ESRI-Taxi')

dfs = []

#####################################################
def read_big_csv():
    fnames = os.listdir("bigdata")
    percentage = 0.99        # Percentage to drop
    for i in range(len(fnames)):
        n = sum(1 for line in open("bigdata/"+fnames[i]))
        skip = sorted(random.sample(range(n), int(n*percentage)))
        dfs.append(pd.read_csv("bigdata/"+fnames[i], usecols = [1,7,8,3], header = None, skip_blank_lines=True, skiprows = skip).dropna(axis=0, how='any'))


read_big_csv()
df1 = pd.concat(dfs, ignore_index=True)
df1 = df1[[1,7,8,3]]
# Concatenate all data into one DataFrame
df1.columns = ['datetime', 'PUloc', 'DOloc', 'Count']

#####################################################

def read_small_csv():
    fnames = os.listdir("data")
    percentage = 0        # Percentage to drop
    for i in range(len(fnames)):
        n = sum(1 for line in open("data/"+fnames[i]))
        skip = sorted(random.sample(range(n), int(n*percentage)))
        dfs.append(pd.read_csv("data/"+fnames[i], usecols = [1,5,6,7], header = None, skip_blank_lines=True, skiprows = skip).dropna(axis=0, how='any'))
        #dfs.append(pd.read_csv("data/"+fnames[i], usecols = [1,5,6,7], header = None, skip_blank_lines=True).dropna(axis=0, how='any').sample(frac = frac))


read_small_csv()
# Concatenate all data into one DataFrame
df1 = pd.concat(dfs, ignore_index=True)
df1.columns = ['datetime', 'PUloc', 'DOloc', 'Count']
###############################################################################
## Validation for NaN entries ##
################################
print("Validation for Null entries:")
print("PUtime empty count :", df1['datetime'].isnull().sum())
print("PUloc empty count :", df1['PUloc'].isnull().sum())
print("DOloc empty count :", df1['DOloc'].isnull().sum())
print("Count empty count :", df1['Count'].isnull().sum())
###############################################################################
## Explore data statistics ##
#############################
df1.describe()

print("Unique Pick-UP locations:",len(df1.PUloc.unique()))
print("Unique Drop-off locations:",len(df1.DOloc.unique()))
###############################################################################
## Handling Time ##
###################
df2 = df1[(df1['datetime'].str.len() < 20)]
hours_range = 8

df2['datetime'] = pd.to_datetime(df2['datetime'])

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
df4 = df3.astype('int')
df5 = df4.groupby(['PUloc','DOloc','year','month','day','day_part','dayofweek','week'])['Count'].sum().reset_index()
df6 = df5[(df5['Count'] > 10)]

#df6 = df4.drop(['year'],axis = 1)
#df7 = df4.drop(['year','week'],axis = 1)
df10 = df6.groupby(['PUloc','DOloc']).size().reset_index(name='Count').sort_values('Count', ascending=False).head(10)
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
    print("DataSet: {0}\tModel: {1}\t\tR2 train = {2:.3f}\tR2 test = {3:.3f}".format(name, str(model).split('(')[0], R2, R2_test))

    for est in [100,300]:
        for nodes in [500, 1000, 2000]:
                ##########################
                ### Random forst Model ###
                ##########################
                model = RandomForestRegressor(n_estimators = 200, random_state = 0, max_leaf_nodes = nodes)
                model.fit(X_train,Y_train)
                R2 = model.score(X_train,Y_train)
                R2_test = model.score(X_test,Y_test)
                print("DataSet: {0}\tModel: {1}\t\t# of trees = {4} \tnodes = {5}\tR2 train = {2:.3f}\tR2 test = {3:.3f}".format(name, str(model).split('(')[0], R2, R2_test, est,nodes))
        
                ###############################
                ### Gradient Boosting Model ###
                ###############################
        #        model = GradientBoostingRegressor(n_estimators= est, learning_rate=0.01, random_state=0, loss='ls')
        #        model.fit(X_train,Y_train)
        #        R2 = model.score(X_train,Y_train)
        #        R2_test = model.score(X_test,Y_test)
        #        print 'DataSet: {0}\tModel: {1}\t# of trees = {4}\tR2 train = {2:.3f}\tR2 test = {3:.3f}'.format(name, str(model).split('(')[0], R2, R2_test, est)

###############################################################################
run_regression_split('df6',df6)
print("\n")
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
    print('DataSet: {0}\tModel: {1}\t\tR2 = {2:.3f}'.format(name, str(model).split('(')[0], R2))

    for est in [10, 200]:
        
        ##########################
        ### Random forst Model ###
        ##########################
        model = RandomForestRegressor(n_estimators = est, random_state = 0)
        model.fit(X,Y)
        R2 = model.score(X,Y)
        print('DataSet: {0}\tModel: {1}\t\tR2 = {2:.3f}\t# of trees = {3}'.format(name, str(model).split('(')[0], R2, est))
        
        ###############################
        ### Gradient Boosting Model ###
        ###############################
        model = GradientBoostingRegressor(n_estimators= est, learning_rate=0.01, random_state=0, loss='ls')
        model.fit(X,Y)
        R2 = model.score(X,Y)
        print('DataSet: {0}\tModel: {1}\tR2 = {2:.3f}\t# of trees = {3}'.format(name, str(model).split('(')[0], R2, est))

###############################################################################
run_regression('df6',df6)
print("\n")
###############################################################################
### Random forst Model ###
##########################
df = pd.read_csv("df.csv")

X = df6.iloc[:,0:-1]
Y = df6.iloc[:,-1]
Y = Y.values.reshape(len(X))

model = RandomForestRegressor(n_estimators = 200, random_state = 0)
model.fit(X,Y)


R2 = model.score(X,Y)

h = model.predict(X)

## at this point Y and h should be arrays:
mse = np.mean((Y - h)**2)

print 'Model: {0}\t\tR2 = {1:.4f}\tmse = {2:.4f}'.format(str(model).split('(')[0], R2, mse)

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
## Plotting the Actual Vs Predicted ##
import pylab as pl


h = model.predict(X)
h = pd.DataFrame(h, columns=['Predicted'])
y = pd.DataFrame(Y)
diff = pd.concat(y,h, axis =1)
diff.columns = [['act','predicted']]
diff['residual'] = diff['act'] - diff['predicted']
plt.hist(diff['residual'], bins=100, range = (0,1))
pl.scatter(y, h)
pl.plot(np.arange(0, 1000), np.arange(0, 1000), label="r^2=" + str(R2), c="r")
pl.legend(loc="lower right")
pl.title("RandomForest Regression with scikit-learn")
pl.show()