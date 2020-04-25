from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# following tutorial:
# https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
def cluster_on_PCA(finalDf):
    X = np.array([finalDf['principal component 1'],finalDf['principal component 2']]).T
    k=5
    km = KMeans(n_clusters=k)
    y_km = km.fit_predict(X)

    # plot the 5 clusters
    fig = plt.figure()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    colors = ['midnightblue','yellowgreen', 'cyan', 'r', 'gray']
    for i in range(k):
        plt.scatter(
            X[y_km == i, 0], X[y_km == i, 1],
            s=50, c=colors[i],
            marker='o', edgecolor='black',
            label='cluster '+str(i+1)
        )
    #plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='*',
        c='yellow', edgecolor='black',
        label='centroids'
    )
    plt.title("Kmeans clustering performed on the reduced data")
    plt.legend(scatterpoints=1, frameon=True)
    plt.grid()
    plt.show()
