'''
Code by: Milan Bidare and Juan Antonio Martinez
CME250: Introduction to Machine Learning
Part II: Prediction & Selection
'''
from mySVM import mySVM
from myDTR import myDTR
from lin_reg import lin_reg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
# from remove_outliers import remove_outliers
# from PCA import PCA
# from plot_PCA import plot_PCA
# from cluster_on_PCA import cluster_on_PCA

if __name__ == "__main__":
    # Loading required data prettily
    url = "movie_metadata.csv"
    col = ['num_critic_for_reviews', 'duration', 'director_facebook_likes',
       'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',
       'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',
       'num_user_for_reviews', 'budget', 'title_year',
       'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio',
       'movie_facebook_likes']
    df = pd.read_csv(url, skipinitialspace=True, usecols=col)
    # df.rename(
    #     columns={
    #         "num_critic_for_reviews": "num_ reviews",
    #         "facenumber_in_poster": "num_faces",
    #         "movie_facebook_likes": "fb_likes",
    #     },
    #     inplace=True
    # )

    # Drop all rows with NaN elements
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # features = list(df.columns)
    # features.remove("imdb_score")

    # Training, validation, and testing sets
    x = df.drop(['imdb_score'],axis=1)
    y = df['imdb_score']
    x_trainval, x_test, y_trainval, y_test = train_test_split(x, y, random_state=0, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x_trainval,y_trainval, random_state=0, test_size=0.25)

    # Linear Regression
    LinModel = lin_reg(x_train, y_train, x_val, y_val)
    print("Linear Regression Test Accuracy:", LinModel.score(x_test, y_test))
    
    # mySVM(x_train, x_val, y_train, y_val)
    
    # Decision Tree Regression (DTR)
    train_score_DTR, val_score_DTR, ccp_alpha = myDTR(x_train, x_val, y_train, y_val)
    DTRmodel = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
    DTRmodel.fit(x_train, y_train)
    print("DTR Accuracy on train:{}".format(DTRmodel.score(x_train,y_train)))
    print("DTR Accuracy on val:{}".format(DTRmodel.score(x_val, y_val)))
    #print("DTR Accuracy on trainval:{}".format(DTRmodel.score(x_trainval,y_trainval)))
    print("DTR Accuracy on test:{}".format(DTRmodel.score(x_test,y_test)))
    
    # predict on train set
    plt.figure()
    train_predict = DTRmodel.predict(x_train)
    train_predict = np.round(train_predict, 1)
    plt.scatter(y_train, train_predict, s=10)
    x = np.linspace(2, 10, 1000)
    plt.plot(x,x,'red');
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("DTR Performance on Train Set")
    plt.grid()
    
    # predict on validation set
    plt.figure()
    val_predict = DTRmodel.predict(x_val)
    val_predict = np.round(val_predict, 1)
    plt.scatter(y_val, val_predict, s=10)
    x = np.linspace(2, 10, 1000)
    plt.plot(x,x,'red');
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("DTR Performance on Validation Set")
    plt.grid()
  
    
    