'''
Code by: Milan Bidare and Juan Antonio Martinez
CME250: Introduction to Machine Learning
Part II: Prediction & Selection
Decision Tree Regressor
'''
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

def myDTR(x_train,x_val,y_train,y_val):
    model = DecisionTreeRegressor(random_state=0).fit(x_train, y_train)
    path = model.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # The last tree is one 1 node, we remove it
    ccp_alphas = ccp_alphas[:-1]
    impurities = impurities[:-1]

    # Train a tree for each alpha
    ms = []
    print("Total to Load = {}".format(len(ccp_alphas)))
    for i,ccp_alpha in enumerate(ccp_alphas):
        # if i%2==0 or i%3 == 0 or i%5 == 0 or i%7 == 0:
        #     continue
        if i%101 == 0:
            print("Loading: " + str(i))
        m = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
        m.fit(x_train, y_train)
        ms.append(m)
    train_scores = [m.score(x_train, y_train) for m in ms]
    val_scores = [m.score(x_val, y_val) for m in ms]
    
    ## Identify the best hyperparameters for accuracy
    # print("MAXtrain Accuracy Training:{}".format(max(train_scores)))
    maxtrainidx = train_scores.index(max(train_scores))
    # print("MAXtrain Accuracy Validation:{}".format(val_scores[maxtrainidx]))
    # print("MAXtrain ccp_alphas:{}".format(ccp_alphas[maxtrainidx]))
    # print("MAXtrain index:{}\n".format(maxtrainidx))

    # print("MAXval Accuracy Validation:{}".format(max(val_scores)))
    maxvalidx = val_scores.index(max(val_scores))
    # print("MAXval Accuracy Training:{}".format(train_scores[maxvalidx]))
    # print("MAXval ccp_alphas:{}".format(ccp_alphas[maxvalidx]))
    # print("MAXval index:{}\n".format(maxvalidx))

    # Plot accuracy of train and val scores for each cpp_alpha
    # plt.scatter(val_scores,train_scores)
    # plt.show()

    # This is probably the worst way to do this
    sum_scores = []
    for j in range(len(val_scores)):
        sum_scores.append(val_scores[j]+train_scores[j])
    maxsumidx = sum_scores.index(max(sum_scores))
    # print("MAXsum Accuracy Validation:{}".format(val_scores[maxsumidx]))
    # print("MAXsum Accuracy Training:{}".format(train_scores[maxsumidx]))
    # print("MAXsum ccp_alphas:{}".format(ccp_alphas[maxsumidx]))
    # print("MAXsum index:{}\n".format(maxsumidx))

    # returning best cpp_alpha hyperparameter
    return train_scores[maxvalidx], val_scores[maxvalidx], ccp_alphas[maxvalidx]


