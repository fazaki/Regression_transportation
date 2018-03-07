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
from sklearn import preprocessing 
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
hours_range = 8
df2 = df1[(df1['datetime'].str.len() < 20)]
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
    ax = sns.countplot(x=col, data=df, palette = ['lightblue'])
    if n == "w": 
        ax.axes.set_xticklabels(["MON", "TUE","WED","THU","FRI","SAT","SUN"])
        ax.set_title("Weekly Pattern distribution",fontsize=24)
        ax.set_xlabel("Day of Week",fontsize=20)
        ax.set_ylabel("Trips Count",fontsize=20)
    elif n == "m":
        ax.axes.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
        ax.set_title("Monthly Pattern distribution",fontsize=24)
        ax.set_xlabel("Month",fontsize=20)
        ax.set_ylabel("Trips Count",fontsize=20)
hst(df9,'dayofweek','w')

fig, ax = plt.subplots(figsize=(10,6))
ax = sns.barplot('Route_ID', y='Count', data=df11, palette = "Blues_d", order = df11.Route_ID,)
ax.set_title("Routes Pattern Barplot",fontsize=24)
ax.set_xlabel("Route ID",fontsize=20)
ax.set_ylabel("People Count",fontsize=20)

fig, ax = plt.subplots(figsize=(10,6))
ax = sns.countplot(x=col, data=df, palette = ['blue'])
ax.axes.set_xticklabels(["MON", "TUE","WED","THU","FRI","SAT","SUN"])
###############################################################################
## Data Filtering on certain trajectory ##
##########################################
df3 = df2.drop(['datetime'],axis = 1)
df3 = df3[['PUloc','DOloc','year','month','day','day_part','dayofweek','week','Count']]
df3.Count = df3.Count.astype('int')
df5 = df3.groupby(['PUloc','DOloc','year','month','day','day_part','dayofweek','week'])['Count'].sum().reset_index()
df6 = df5[(df5['Count'] > 10)]
df7 = df6.drop(['week'],axis = 1)          ## removing the week number feature
df8 = df7.copy()
df8.PUloc = df8.PUloc.astype('str')
df8.DOloc = df8.DOloc.astype('str')
df8['Route'] = df8.PUloc + "->" + df8.DOloc
le = preprocessing.LabelEncoder()
le.fit(df8.Route)
df8['Route_ID'] = le.transform(df8.Route)
df9 = df8.drop(['PUloc','DOloc','Route'],axis = 1)
df9 = df9[['Route_ID','year','month','day','day_part','dayofweek','Count']]
df10 = df9.groupby(['Route_ID']).size().reset_index(name='Count').sort_values('Count', ascending=False).head(10)

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
    #model = DecisionTreeRegressor()
    #model.fit(X_train,Y_train)
    #R2 = model.score(X_train,Y_train)
    #R2_test = model.score(X_test,Y_test)
    #print("DataSet: {0}\tModel: {1}\t\tR2 train = {2:.3f}\tR2 test = {3:.3f}".format(name, str(model).split('(')[0], R2, R2_test))

    for est in [50]:
        for nodes in [2000]:
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
                #model = GradientBoostingRegressor(n_estimators = 200, random_state = 0, max_leaf_nodes = nodes, learning_rate=0.01, loss='ls')
                #model.fit(X_train,Y_train)
                #R2 = model.score(X_train,Y_train)
                #R2_test = model.score(X_test,Y_test)
                #print("DataSet: {0}\tModel: {1}\t\t# of trees = {4} \tnodes = {5}\tR2 train = {2:.3f}\tR2 test = {3:.3f}".format(name, str(model).split('(')[0], R2, R2_test, est,nodes))
        
###############################################################################
run_regression_split('df7',df7)
print("\n")
###############################################################################
###############################################################################
### Random forst Model ###
##########################
df = pd.read_csv("df.csv")

X = df9.iloc[:,0:-1]
Y = df9.iloc[:,-1]
Y = Y.values.reshape(len(X))

model = RandomForestRegressor(n_estimators = 50, max_leaf_nodes = 2000,  random_state = 0)
model.fit(X,Y)


R2 = model.score(X,Y)

h = model.predict(X)

## at this point Y and h should be arrays:
mse = np.mean((Y - h)**2)

print('Model: {0}\t\tR2 = {1:.4f}\tmse = {2:.4f}'.format(str(model).split('(')[0], R2, mse))

###############################################################################
### PREDICTION ###
##################

Route_ID = 2680
year = 2018
month = 6
day = 10
day_part = 0
dayofweek = 6

input_ = [(Route_ID ,year ,month ,day ,day_part ,dayofweek)]

math.ceil(model.predict(input_)[0])

###############################################################################
## Plotting the Actual Vs Predicted ##
import pylab as pl


h = model.predict(X)
h = pd.DataFrame(h, columns=['Predicted'])
y = pd.DataFrame(Y)
diff = pd.Dataframe(y[[0],h[[0]])
diff.columns = [['act','predicted']]
diff['residual'] = diff['act'] - diff['predicted']
plt.hist(diff['residual'], bins=100, range = (0,1))
pl.scatter(y, h)
pl.plot(np.arange(0, 1000), np.arange(0, 1000), label="r^2=" + str(R2), c="r")
pl.legend(loc="lower right")
pl.title("RandomForest Regression with scikit-learn")
pl.show()