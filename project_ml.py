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
dat['sales'] = dat.groupby('id')['sales'].apply(lambda x: x.interpolate().bfill().ffill())
dat['va'] = dat.groupby('id')['va'].apply(lambda x: x.interpolate().bfill().ffill())

# Applying the same treatment to the others variables except id, year and yearest
cols_to_impute = dat.columns.difference(['id', 'year', 'yearest']) 
for col in cols_to_impute: 
    dat[col] = dat.groupby('id')[col]. apply (lambda x: x.ffill().bfill())

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


dat.gom.describe() # à voir
dat = dat.sort_values(by='gom', ascending=True)
dat = dat.iloc[9:]

dat = dat.sort_values(by='Unnamed: 0', ascending=True)


age = dat.year - dat.yearest
dat.insert(4, 'age', age)

