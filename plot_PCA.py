# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:48:34 2020

@author: Admin
"""

def plot_PCA(finalDf, xaxis, yaxis):
    import matplotlib.pyplot as plt
    scores = [0, 2, 4, 6, 8]
    colors = ['midnightblue','yellowgreen', 'cyan', 'r', 'gray']
    alpha = [1,0.8,0.6,0.2,0.4]
    for minscore, color,alp in zip(scores,colors,alpha):
        maxscore = minscore+2
        indices_smaller = finalDf['imdb_score'].le(maxscore)  
        indices_greater = finalDf['imdb_score'].gt(minscore)
        plot_indices =  indices_smaller & indices_greater
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        plt.xlim(xaxis)
        plt.ylim(yaxis)
        ax.set_title('2 Component PCA for IMDB Score between ' + str(minscore) + ' and ' + str(maxscore), fontsize = 20)
        ax.grid()
        ax.scatter(finalDf.loc[plot_indices, 'principal component 1'],
                   finalDf.loc[plot_indices, 'principal component 2'], s=10, c=color, alpha=alp)

    # Visualizing Data: PCA components with different ratings (try plotting each in a different graph)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 Component PCA', fontsize = 20)

    # All together
    for minscore, color, alp in zip(scores, colors, alpha):
       maxscore = minscore+2
       indices_smaller = finalDf['imdb_score'].le(maxscore)
       indices_greater = finalDf['imdb_score'].ge(minscore)
       indicesToKeep =  indices_smaller & indices_greater
       ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                 , finalDf.loc[indicesToKeep, 'principal component 2']
                 , c = color
                 , s = 5, alpha = alp)
    ax.legend(['0-2','2-4','4-6','6-8','8-10'],title='IMDB Scores', frameon=True)
    ax.grid()
        
    # Visualizing Data: PCA component 1 vs IMDB score
    fig2 = plt.figure(figsize = (8,8))
    ax2 = fig2.add_subplot(1,1,1) 
    ax2.set_xlabel('Principal Component 1', fontsize = 15)
    ax2.set_ylabel('IMDB Score', fontsize = 15)
    ax2.set_title('Principal Component 1 vs IMDB Score', fontsize = 20)
    ax2.grid()
    plt.scatter(finalDf['principal component 1'], finalDf['imdb_score'], s=10)
    
    # Visualizing Data: PCA component 2 vs IMDB score
    fig3 = plt.figure(figsize = (8,8))
    ax3 = fig3.add_subplot(1,1,1) 
    ax3.set_xlabel('Principal Component 2', fontsize = 15)
    ax3.set_ylabel('IMDB Score', fontsize = 15)
    ax3.set_title('Principal Component 2 vs IMDB Score', fontsize = 20)
    ax3.grid()
    plt.scatter(finalDf['principal component 2'], finalDf['imdb_score'], s=10)
    plt.show()