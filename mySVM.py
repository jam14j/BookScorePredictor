'''
Code by: Milan Bidare and Juan Antonio Martinez
CME250: Introduction to Machine Learning
Part II: Prediction & Selection
Support Vector Machine
following this tutorial:
https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
'''
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
import numpy as np

def mySVM(x_train,x_val,y_train,y_val):
    score_train = -100
    score_val = -100
    score_sum = -100
    for c in range(1,1000,10):
        print(c)
        model = svm.SVR(C=c/100.0).fit(x_train,(y_train*10).astype('int'))
        st = model.score(x_train, (y_train*10).astype('int'))
        score_train = st if st>score_train else score_train
        sv = model.score(x_val, (y_val*10).astype('int'))
        score_val = sv if sv > score_val else score_val
        ss = st+sv
        score_sum = ss if ss > score_sum else score_sum

    print("Max Accuracy Training:{}".format(score_train))
    print("Max Accuracy Validation:{}".format(score_val))

