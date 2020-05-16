'''
Code by: Milan Bidare and Juan Antonio Martinez
CME250: Introduction to Machine Learning
Part II: Prediction & Selection
Support Vector Machine
'''
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

def mySVM(x_train,x_val,y_train,y_val):
    score_train = -100
    train_list = []
    score_val = -100
    val_list = []
    score_sum = -100
    opt_c = -100
    maxidx = -100
    for c in range(1,1000,10):
        print(c)
        model = svm.SVR(C=c/100.0)
        model.fit(x_train,(y_train*10).astype('int'))
        st = model.score(x_train, (y_train*10).astype('int'))
        score_train = st if st>score_train else score_train
        sv = model.score(x_val, (y_val*10).astype('int'))
        score_val = sv if sv > score_val else score_val
        ss = st+sv
        val_list.append(sv)
        train_list.append(st)
        if ss > score_sum:
            score_sum = ss
            opt_c = c
            maxidx = (int) ((c-1)/10)
    print("C:{}".format(opt_c))
    print("maxidx:{}".format(maxidx))
    print("Accuracy Validation:{}".format(val_list[maxidx]))
    print("Accuracy Training:{}".format(train_list[maxidx]))
    model = svm.SVR(C=opt_c/100.0)
    model.fit(x_train, (y_train * 10).astype('int'))
    yval_predict = model.predict(x_val)
    ytrain_predict = model.predict(x_train)

    plt.figure()
    plt.scatter(y_train, ytrain_predict / 10.0, s=10)
    plt.plot(y_train, y_train, color="red")
    plt.title("SVM Performace on Train Set")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.grid()
    plt.figure()
    plt.scatter(y_val, yval_predict/10.0, s=10)
    plt.plot(y_val,y_val, color="red")
    plt.title("SVM Performace on Validation Set")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.grid()
    plt.show()


