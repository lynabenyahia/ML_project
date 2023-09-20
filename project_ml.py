#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:31:56 2023

@author: lyna
"""
    

import pandas as pd
import matplotlib

dat = pd.read_csv('/Users/lyna/Desktop/M2/supervised learning/data_assignment_1.csv')    
dat = dat.drop(['Unnamed: 0'], axis=1)
    
# =============================================================================
# NA TREATMENT
# =============================================================================

# Replace the year by the min for the id
dat['yearest'] = dat.groupby('id')['yearest'].transform(lambda x: x.min())
# Replace remaining missing values by the mode of the column
dat['yearest'].fillna(dat['yearest'].mode()[0], inplace=True)

# Replace sales and va missing by the mean of the value before and after the value
dat['sales'] = dat.groupby('id')['sales'].apply(lambda x: x.interpolate().bfill().ffill()).reset_index(drop=True)
dat['va'] = dat.groupby('id')['va'].apply(lambda x: x.interpolate().bfill().ffill()).reset_index(drop=True)

# Applying the same treatment to the others variables except id, year and yearest
cols_to_impute = dat.columns.difference(['id', 'year', 'yearest']) 
for col in cols_to_impute: 
    dat[col] = dat.groupby('id')[col]. apply (lambda x: x.ffill().bfill()).reset_index(drop=True)

# Use median imputation for vartables with remaining missing values
cols_to_median_impute = ['ipnm', 'ipnf', 'ipnc', 'enggrad', 'va', 'gom', 'reext', 'rdint', 'ipr', 'patent', 'sales']
for col in cols_to_median_impute:
    median_value = dat[col].median() 
    dat[col].fillna(median_value, inplace=True)

missing_values_final = dat.isnull().mean() * 100
missing_values_final.sort_values(ascending=False)

# =============================================================================
# OUTLIERS TREATMENT
# =============================================================================
matplotlib.pyplot.boxplot(dat.patent)
matplotlib.pyplot.boxplot(dat.reext)
matplotlib.pyplot.boxplot(dat.rdint)
matplotlib.pyplot.boxplot(dat.enggrad)
matplotlib.pyplot.boxplot(dat.pertot)
matplotlib.pyplot.boxplot(dat.gom)
matplotlib.pyplot.boxplot(dat.va)
matplotlib.pyplot.boxplot(dat.sales)


# Delete the 2 outliers in the gom column, which was extremly high comparing to the rest of the column
dat.gom.describe() # à voir
dat = dat.sort_values(by='gom', ascending=False)
dat = dat.iloc[2:]

# Re sorting the database
dat = dat.sort_values(['id', 'year'],
              ascending = [True, True])

# Creating an age variable
age = dat.year - dat.yearest
dat.insert(4, 'age', age)

# Creating a HGF variable
threshold = dat['sales'].quantile(0.9)
dat['hgf'] = dat['sales'] > threshold
dat['hgf'] = dat['hgf'].astype(int)

# =============================================================================
# Which company is going to be a HGF in the last of year of the sample ?
# =============================================================================





