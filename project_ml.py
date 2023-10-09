#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:31:56 2023

@author: Lyna, Jeanne, Marion
"""
    

import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


dat = pd.read_csv('/Users/lyna/Desktop/M2/supervised learning/data_assignment_1.csv')    

#dat = pd.read_csv('C:/Users/epcmic/OneDrive/Documents/GitHub/ML_project/data_assignment_1.csv')


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
dat['9th_percentile'] = dat.groupby('year')['sales'].transform(lambda x: x.quantile(0.9))
dat['hgf'] = (dat['sales'] > dat['9th_percentile']).astype(int)
dat.drop(columns=['9th_percentile'], inplace=True)

# Creating a R&D intensity variable
dat['RD_intensity'] = ( (dat.rdint + dat.reext) / dat.va ) * 100

# =============================================================================
# Which company is going to be a HGF in the last of year of the sample ?
# =============================================================================

# Creating a training set and a test set 
dat_training = dat[dat['year'] < 2012]
dat_test = dat[dat['year'] == 2012]

# Creating X_train, X_test, y_train et y_test
features = ['age', 'pertot', 'enggrad', 'sales', 'va', 'gom', 'rdint', 'reext', 'ipnc', 'ipnf', 'ipnm', 'ipr', 'patent', 'RD_intensity']

X_train = dat_training[features]
X_test = dat_test[features]
y_train = dat_training['hgf']
y_test = dat_test['hgf']

# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train, y_train)

print("Accuracy (training): {:.3f}".format(logreg.score(X_train, y_train)))
print("Accuracy (test): {:.3f}".format(logreg.score(X_test, y_test)))


# =============================================================================
# SUPPORT VECTOR MACHINES
# =============================================================================
svc = SVC()

# Fit the model on the training set 
svc.fit(X_train, y_train)

print("Accuracy (training): {:.3f}".format(svc.score(X_train, y_train)))
print("Accuracy (test): {:.3f}".format(svc.score(X_test, y_test)))

plt.boxplot(X_train, manage_ticks=False)
plt.yscale("symlog")
plt.xlabel("Feature index")
plt.ylabel("Feature value")

# We compute the minimum value for each feature on the training set
min_on_training = X_train.min(axis=0)
# And the range of each feature (max - min) on the training set
range_on_training = (X_train - min_on_training).max(axis=0)

# Then we subtract the min and divide by range
X_train_scaled = (X_train - min_on_training) / range_on_training
print("Minimum for each feature\n", X_train_scaled.min(axis=0))
print("Maximum for each feature\n", X_train_scaled.max(axis=0))

# Now we obtain the same scale!
plt.boxplot(X_train_scaled, manage_ticks=False)
plt.yscale("symlog")
plt.xlabel("Feature index")
plt.ylabel("Feature value")

# Make sure to use the same scaling on the test setm, using min and range of the training set
X_test_scaled = (X_test - min_on_training) / range_on_training

# Finally, check the performance of the model
svc = SVC()
svc.fit(X_train_scaled, y_train)

print("Accuracy (training): {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy (test): {:.3f}".format(svc.score(X_test_scaled, y_test)))
    

# =============================================================================
# K NEAREST NEIGHBORS
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Instantiate the model   
knn = KNeighborsClassifier(n_neighbors = 19)
# Fit the model using the training set 
knn.fit(X_train, y_train)
# Make prediction on the test set
knn.predict(X_test)
y_pred = knn.predict(X_test)

# Check the performance of the model
print('Confusion matrix: \n', confusion_matrix(y_test, y_pred))
print('Classifier performance: \n', classification_report(y_test, y_pred, digits=3))


# Selecting the optimal value of K with `GridSearchCV()`

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

# The function offers a much easier way to select the tuning parameters

# create a dictionary of all K values we want to test 
param_grid = {'n_neighbors': np.arange(1, 20)}

# Initiate the model
knn_gscv = GridSearchCV(KNeighborsRegressor(), param_grid, cv=10)

# Fit the model on the training set 
knn_gscv.fit(X_train, y_train)

# Check the best model 
knn_gscv.best_params_