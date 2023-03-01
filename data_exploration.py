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
train.shape

# Top 5 cols
train.head()

# List all cols
train.columns

# list null value count in each cols
train.isnull().sum()

# list non-null value count in each cols
train.info()

# list null value percentage in each cols
train.isnull().mean() * 100

