### Package imports ###
# ------------------- #
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


################ PCA REDUCTION COMPLETE (MOSTLY) ###########################
##### date and ticker labels are not being transferred atm ###############


### Dimensionality Reduction using PCA ###
# -------------------------------------- #
def pca_reduction(df):
    
    print("\nReducing data size for each stock using PCA...")

    # transform nasdaq values into pandas dataframe
    df = pd.DataFrame(df)

    scaled_df = pd.DataFrame(StandardScaler().fit_transform(df))  # standardise/scale data
    pca = PCA(n_components = 10)  # set number of values each data field should be reduced to
    # perform pca on scaled data
    pca.fit(scaled_df)
    data_pca = pca.transform(scaled_df)
    pca_df = pd.DataFrame(data_pca,index=df.index[:])  # add stock labels to reduced df

    # export reduced dataframe to csv file for analysis
    pca_df.to_csv('reduced.csv', mode="w")

    print("PCA Data Reduction completed ✓")

    return pca_df



### KMeans clustering is giving some weird results and i am still trying to wrap my head around the returned clusters ###
def kmeans(df):

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=4)
    kmeans_labels = kmeans.fit(df)
    
    # print(df.index[x], kmeans_labels[x])

    # Print the cluster labels
    print(kmeans_labels.labels_)

    return df
