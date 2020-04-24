# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:34:07 2020
Following Tutorial: https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Data_Visualization_Iris_Dataset_Blog.ipynb
@author: Admin
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('seaborn-white')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Loading required data prettily
# url = "C:/Users/mibid/Documents/GitHub/BookScorePredictor/movie_metadata.csv"
# col = ['num_critic_for_reviews', 'duration','gross','facenumber_in_poster','budget','movie_facebook_likes','imdb_score']
# df = pd.read_csv(url, skipinitialspace=True, usecols=col)
# df.rename(
#     columns={
#         "num_critic_for_reviews": "num_reviews",
#         "facenumber_in_poster": "num_faces",
#         "movie_facebook_likes": "fb_likes",
#     },
#     inplace=True
# )
# df.dropna(inplace = True)
# df.reset_index(drop=True,inplace=True)
# features = list(df.columns)
# features.remove("imdb_score")

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
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

gradient = 10
scores = np.linspace(1,10,gradient)
#colors = cm.rainbow(np.linspace(0, 1, gradient))
colors = ['r','orange','yellowgreen', 'green', 'cyan', 'steelblue', 'midnightblue', 'mediumpurple','violet','black']
for minscore, maxscore, color in zip(scores,scores[1:], colors):
   indices_smaller = finalDf['imdb_score'].le(maxscore)  
   indices_greater = finalDf['imdb_score'].ge(minscore)
   indicesToKeep =  indices_smaller & indices_greater
   ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
             , finalDf.loc[indicesToKeep, 'principal component 2']
             , c = color
             , s = 5)
ax.legend(scores)
ax.grid()

# Visualizing Data: PCA component 1 vs IMDB score
fig2 = plt.figure(figsize = (8,8))
ax2 = fig2.add_subplot(1,1,1) 
ax2.set_xlabel('Principal Component 1', fontsize = 15)
ax2.set_ylabel('IMDB Score', fontsize = 15)
ax2.set_title('Principal Component 1 vs IMDB Score', fontsize = 20)
plt.scatter(finalDf['principal component 1'], finalDf['imdb_score'], s=10)

# Visualizing Data: PCA component 2 vs IMDB score
fig3 = plt.figure(figsize = (8,8))
ax3 = fig3.add_subplot(1,1,1) 
ax3.set_xlabel('Principal Component 2', fontsize = 15)
ax3.set_ylabel('IMDB Score', fontsize = 15)
ax3.set_title('Principal Component 2 vs IMDB Score', fontsize = 20)
plt.scatter(finalDf['principal component 2'], finalDf['imdb_score'], s=10)

# Variance
print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))

## There are some outliers in the data where principal component 1 is larger than abs(10). 
## Remove those points and try again to see if the spread is better
indices_outlier = finalDf['principal component 2'].le(-10) 
newdf = df.drop(index = indices_outlier[indices_outlier].index)
newdf.reset_index(drop=True,inplace=True)
indices_outlier = finalDf['principal component 2'].ge(10)
newdf = newdf.drop(index = indices_outlier[indices_outlier].index)
newdf.reset_index(drop=True,inplace=True)

df = newdf
