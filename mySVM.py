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
    # score_train = -100
    # score_val = -100
    # for c in range(1,1000,10):
    #     print(c)
    #     model = svm.SVR(C=c/100.0).fit(x_train,(y_train*10).astype('int'))
    #     st = model.score(x_train, (y_train*10).astype('int'))
    #     score_train = st if st>score_train else score_train
    #     sv = model.score(x_val, (y_val*10).astype('int'))
    #     score_val = sv if sv > score_val else score_val
    # print("Max Accuracy Training:{}".format(score_train))
    # print("Max Accuracy Validation:{}".format(score_val))

    model = DecisionTreeRegressor(random_state=0).fit(x_train, y_train)
    path = model.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # The last tree is one 1 node, we remove it
    ccp_alphas = ccp_alphas[:-1]
    impurities = impurities[:-1]

    # Train a tree for each alpha
    ms = []
    print("Total = {}".format(len(ccp_alphas)))
    for i,ccp_alpha in enumerate(ccp_alphas):
        if i%2 == 0 or i%3 == 0 or i%5 == 0:
            continue
        if i%101 == 0:
            print(i)
        m = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
        m.fit(x_train, y_train)
        ms.append(m)
    train_scores = [1 - m.score(x_train, y_train) for m in ms]
    val_scores = [1 - m.score(x_val, y_val) for m in ms]
    print("MAXtrain Accuracy Training:{}".format(max(train_scores)))
    maxtrainidx = train_scores.index(max(train_scores))
    print("MAXtrain Accuracy Validation:{}".format(val_scores[maxtrainidx]))
    print("MAXtrain ccp_alphas:{}".format(ccp_alphas[maxtrainidx]))
    print("MAXtrain index:{}\n".format(maxtrainidx))

    print("MAXval Accuracy Validation:{}".format(max(val_scores)))
    maxvalidx = val_scores.index(max(val_scores))
    print("MAXval Accuracy Training:{}".format(train_scores[maxvalidx]))
    print("MAXval ccp_alphas:{}".format(ccp_alphas[maxvalidx]))
    print("MAXval index:{}\n".format(maxvalidx))

    maxsumidx = np.sum(np.array(val_scores),np.array(train_scores)).index(max(np.sum(np.array(val_scores),np.array(train_scores))))
    print("MAXsum Accuracy Validation:{}".format(val_scores[maxsumidx]))
    print("MAXsum Accuracy Training:{}".format(train_scores[maxsumidx]))
    print("MAXsum ccp_alphas:{}".format(ccp_alphas[maxsumidx]))
    print("MAXsum index:{}\n".format(maxsumidx))

    # maxsum2idx = val_scores.index(max(0.25*val_scores.astype('int') + 0.75*train_scores.astype('int')))
    # print("MAXsum Accuracy Validation:{}".format(val_scores[maxsum2idx]))
    # print("MAXsum Accuracy Training:{}".format(train_scores[maxsum2idx]))
    # print("MAXsum ccp_alphas:{}".format(ccp_alphas[maxsum2idx]))
    # print("MAXsum index:{}\n".format(maxsum2idx))
    # return

    # score_train = model.score(x_train, y_train)
    # print("Accuracy Training:{}".format(score_train))
    #
    # score_val = model.score(x_val, y_val)
    # print("Accuracy Validation:{}".format(score_val))