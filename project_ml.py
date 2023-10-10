#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:31:56 2023

@author: Lyna, Jeanne, Marion
"""
    

import pandas as pd
import matplotlib
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from xgboost import plot_importance



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

# Subtract the min and divide by range
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

# Check the performance of the model
svc = SVC()
svc.fit(X_train_scaled, y_train)

print("Accuracy (training): {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy (test): {:.3f}".format(svc.score(X_test_scaled, y_test)))
    

# =============================================================================
# K NEAREST NEIGHBORS
# =============================================================================
# Selecting the optimal value of K
param_grid = {'n_neighbors': np.arange(1, 20)}
knn_gscv = GridSearchCV(KNeighborsRegressor(), param_grid, cv=10)

# Fit the model on the training set 
knn_gscv.fit(X_train, y_train)
knn_gscv.best_params_

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

# =============================================================================
# DECISION TREE
# =============================================================================

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix:\n{confusion_matrix(y_train, pred)}\n")
        
    elif not train:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, pred)}\n")


tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)


# =============================================================================
#  RANDOM FOREST
# =============================================================================

# Création d'une instance du classificateur RandomForest
rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)

# Entraînement du modèle sur les données d'entraînement
rf_clf.fit(X_train, y_train)

# Appel de la fonction print_score pour afficher les résultats
print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)


# =============================================================================
# XG BOOST
# =============================================================================
# Création d'une instance du classificateur XGBoost
xgb_clf = XGBClassifier(use_label_encoder=False)

# Entraînement du modèle sur les données d'entraînement
xgb_clf.fit(X_train, y_train)

# Appel de la fonction print_score pour afficher les résultats
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)

# Présentation de l'importance de chaque variable
plt.figure(figsize=(10, 6))
plot_importance(xgb_clf, max_num_features=10) 
plt.show()

# Prédictions
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculer la matrice de confusion
confusion = confusion_matrix(y_test, y_pred)

# Créer une heatmap avec Seaborn
sns.set(font_scale=1.2)  # Ajustez la taille de la police si nécessaire
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non 'hgf'", "'hgf'"], yticklabels=["Non 'hgf'", "'hgf'"])
plt.xlabel('Prédictions')
plt.ylabel('Vraies valeurs')
plt.title('Matrice de confusion')
plt.show()

