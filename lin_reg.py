# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:32:19 2020
https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
@author: Admin
"""

import pandas as pd
import numpy as np
import seaborn as sns

#from remove_outliers import remove_outliers
#from PCA import PCA
#from plot_PCA import plot_PCA
#from cluster_on_PCA import cluster_on_PCA
from sklearn import linear_model
import matplotlib.pyplot as plt

def lin_reg(x_train, y_train, x_val, y_val):
    # train a model
    lm = linear_model.LinearRegression()
    model = lm.fit(x_train, y_train)
    
    # predict on train set
    plt.figure(1)
    train_predict = lm.predict(x_train)
    train_predict = np.round(train_predict, 1)
    plt.scatter(y_train, train_predict, s=10)
    x = np.linspace(2, 10, 1000)
    plt.plot(x,x,'red');
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("LR Performance on Train Set")
    plt.grid()
    print("Linear Regression Train Accuracy:", model.score(x_train, y_train))
           
    # predict on validation set
    plt.figure(2)
    val_predict = lm.predict(x_val)
    val_predict = np.round(val_predict, 1)
    plt.scatter(y_val, val_predict, s=10)
    x = np.linspace(2, 10, 1000)
    plt.plot(x,x,'red');
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("LR Performance on Validation Set")
    plt.grid()
    print("Linear Regression Validation Accuracy:", model.score(x_val, y_val))

    return model
    