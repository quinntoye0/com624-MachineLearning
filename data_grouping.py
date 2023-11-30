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



### KMeans Clustering ###
# --------------------- #
def kmeans(df):

    # perform k-means clustering
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(df)
    kmeans.predict(df)
    kmeans_labels = kmeans.labels_
    
    # set up lists to split tickers into their kmeans clsuters
    c0 = []
    c1 = []
    c2 = []
    c3 = []

    # loops reduced df index (tickers names) and splits kmeans labels with ticker names
    for x, ticker in enumerate(df.index):
        kmeans_label = kmeans_labels[x]  # grabs current iteration kmeans cluster label
        if kmeans_label == 0:
            c0.append(ticker)
        elif kmeans_label == 1:
            c1.append(ticker)
        elif kmeans_label == 2:
            c2.append(ticker)
        else:
            c3.append(ticker)

    # displays clusters
    print(f"\nCluster 1:")
    for ticker in c0:
        print(ticker)
    print(f"\nCluster 2:")
    for ticker in c1:
        print(ticker)
    print(f"\nCluster 3:")
    for ticker in c2:
        print(ticker)
    print(f"\nCluster 4:")
    for ticker in c3:
        print(ticker)

    # Print the cluster labels
    # print(kmeans_labels)

    return kmeans_labels
