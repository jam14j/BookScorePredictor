# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:34:07 2020
Following Tutorial: https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Data_Visualization_Iris_Dataset_Blog.ipynb
@author: Admin
"""
def PCA(df, features):

    import pandas as pd 
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    plt.style.use('seaborn-white')
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from plot_PCA import plot_PCA

    # Standardizing Data
    x = df.loc[:, features].values
    y = df.loc[:,['imdb_score']].values
    x = StandardScaler().fit_transform(x)
    
    # Performing PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, df['imdb_score']], axis = 1)
    

    # Visualizing Data: PCA components with different ratings (try plotting each in a different graph)
    # fig = plt.figure(figsize = (8,8))
    # ax = fig.add_subplot(1,1,1) 
    # ax.set_xlabel('Principal Component 1', fontsize = 15)
    # ax.set_ylabel('Principal Component 2', fontsize = 15)
    # ax.set_title('2 Component PCA', fontsize = 20)
    
    # gradient = 10
    # scores = np.linspace(1,10,gradient)
    #colors = cm.rainbow(np.linspace(0, 1, gradient))
    #colors = ['r','orange','yellowgreen', 'green', 'cyan', 'steelblue', 'midnightblue', 'mediumpurple','violet','black']
    # for minscore, maxscore, color in zip(scores,scores[1:], colors):
    #    indices_smaller = finalDf['imdb_score'].le(maxscore)  
    #    indices_greater = finalDf['imdb_score'].ge(minscore)
    #    indicesToKeep =  indices_smaller & indices_greater
    #    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
    #              , finalDf.loc[indicesToKeep, 'principal component 2']
    #              , c = color
    #              , s = 5)
    # ax.legend(scores)
    # ax.grid()
    
    # Variance
    #print("pca.explained_variance_ratio_[0])
    print("The % of data captured by PCA is " + str(np.sum(pca.explained_variance_ratio_)))
    
    return finalDf
    ## There are some outliers in the data where principal component 1 is larger than abs(10). 
  