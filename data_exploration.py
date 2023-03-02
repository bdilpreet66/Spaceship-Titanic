# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 19:09:16 2023

@author: Dilpreet Singh Brar
"""

import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# sample = pd.read_csv('sample_submission.csv')

###################################
###################################
##### Training Data Analysis ######
###################################
###################################

# Size
print("Shape:",train.shape,"\n")

# Top 10 cols
print("Top 10 Columns")
print(train.head(10))

# List all cols
print("\nColumn Names")
print(train.columns)

# list null value count in each cols
print("\nNumber of null values in each of these columns")
print(train.isnull().sum())

# list non-null value count in each cols
print("\nNumber of non-null values in each of these columns")
print(train.info())

# list null value percentage in each cols
print("\n percentage of null values in each column")
print(train.isnull().mean() * 100)

# Since the missing data is small lets try removing the null values
train_wn = train.dropna()
# we just lost 25% of our data, that's a lot of data and we might not want to loose all tahat data so lets try something else

############ HomePlanet ############
print("Home Planed Consists of these three categories: ", train['HomePlanet'].dropna().unique())
train['HomePlanet'].fillna(method='ffill').fillna(method='bfill')