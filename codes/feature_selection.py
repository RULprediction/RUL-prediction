#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns

# Data preprocessing
training_data = pd.read_csv('train.csv', header=0)
testing_data = pd.read_csv('test.csv', header=0)

training_frame = pd.DataFrame(training_data, columns=training_data.columns)
testing_frame = pd.DataFrame(testing_data, columns=testing_data.columns)

# select feature1-21 from the dataset for feature selection
X = training_frame.iloc[:, 14:35]
X_t = testing_frame.iloc[:, 14:35]
# target column: RUL
y_train = training_frame.iloc[:,8]
y_test = testing_frame.iloc[:,8]

# perform different feature selection methods to get the most
# valuable features

# PCA
def perform_pca(samples, components=20):
    """
    samples: training dataset to perform PCA
    components: number of PCA components needed. Default 20
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=components)
    pca.fit_transform(samples)

    #principalDf = pd.DataFrame(data = pricipalComponents, columns=['PC-1', 'PC-2'])
    #percentiage of variance explained for each components
    print('Explained variance ratio(first two components): %s'%str(pca.explained_variance_ratio_))
    #print(principalDf)

    #finalDf = pd.concat([principalDf, training_frame[['RUL']]], axis = 1)

    print('PCA components:%s'%str(pca.components_))
    plt.figure(1, figsize=(8, 10))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_ratio_, linewidth=5)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio_')  
    # by comparing the absolute value of the weight of each feature,
    # feature 1, 6, 7, 8, 12, and 18 can be selected

#perform_pca(X)

# KBest using chi-square
def perform_kbest(x, y, feature_num):
    """
    Perform a filter method kbest for feature selection.
    x: training dataset
    y: target dataset
    feature_num: number of features need to be selected.
    
    """
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    #feature extraction: Chi-squared test, select best features
    feature = SelectKBest(score_func=chi2, k=feature_num).fit(x, y)
    # #fit = test.fit(X, y)
 
    np.set_printoptions(precision=3)
    print('Score list:', feature.scores_)

    selected_features = feature.transform(x)
    #selected_features_test = feature.transform(X_t)
    print('Selected feature list:',selected_features[0:feature_num, :])

#perform_kbest(X, y_train, 6)# 3, 4, 7, 9, 12, 18 are selected

# REF
def perform_ref(x, y, feature_num):
    """
    x: training dataset
    y: target dataset
    feature_num: number of features need to be selected.
        
    """
    from sklearn.feature_selection import RFE
    from sklearn.feature_selection import RFECV
    from sklearn.svm import SVC

    estimator = SVC(kernel = "linear")
    selector = RFECV(estimator, step=1, cv=5)
    #rfe = RFE(estimator=clf, n_features_to_select=feature_num, step=1)
    selector = selector.fit(x, y)
    #clf =clf.fit(selected_features, y_train)
    print('Chosen best 5 feature by rfe:',X.columns[selector.support_])
