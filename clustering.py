import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

fields =['num_critic_for_reviews', 'duration','gross','facenumber_in_poster',
         'budget','movie_facebook_likes','imdb_score']
df = pd.read_csv("movie_metadata.csv", usecols=fields)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
# following tutorial:
# https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
X = np.array([df['gross'],df['imdb_score']]).T
km = KMeans(n_clusters=3)
y_km = km.fit_predict(X)


# plot the 3 clusters
fig = plt.figure()
plt.xlabel("Gross revenue")
plt.ylabel("IMDB score")
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)
plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)
plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)
#plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()