'''
Code by: Milan Bidare and Juan Antonio Martinez
CME250: Introduction to Machine Learning
Part I: Data exploration
'''
import pandas as pd
from remove_outliers import remove_outliers
from PCA import PCA
from plot_PCA import plot_PCA
from cluster_on_PCA import cluster_on_PCA

if __name__ == "__main__": 
    # Loading required data prettily
    url = "movie_metadata.csv"
    col = ['num_critic_for_reviews', 'duration','gross','facenumber_in_poster','budget','movie_facebook_likes','imdb_score']
    df = pd.read_csv(url, skipinitialspace=True, usecols=col)
    df.rename(
        columns={
            "num_critic_for_reviews": "num_reviews",
            "facenumber_in_poster": "num_faces",
            "movie_facebook_likes": "fb_likes",
        },
        inplace=True
    )
 
    # Drop all rows with NaN elements
    df.dropna(inplace = True)
    df.reset_index(drop=True,inplace=True)
    features = list(df.columns)
    features.remove("imdb_score")
    
    # Run PCA until no more extreme outliers (outside threshold) exist
    plot = True
    while True:
        finalDf = PCA(df, features)
        if plot == True: # Plot initial PCA with all outliers
                plot_PCA(finalDf, (-2,14), (-20,25))
                plot = False
                
        df,num_outliers = remove_outliers(df, finalDf, 10) # Threshold for outlier is 10
        if num_outliers == 0:
            break
    plot_PCA(finalDf, (-4,14), (-4,10)) # Plot final PCA with no outliers
    cluster_on_PCA(finalDf)