### Package imports ###
# ------------------- #
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


################ PCA REDUCTION COMPLETE (MOSTLY) ###########################
##### date and ticker labels are not being transferred atm ###############

def dataGrouping(df):

    ### Dimensionality Reduction using PCA ###
    # -------------------------------------- #
    print("\nReducing data size for each stock using PCA...")

    df = pd.DataFrame(df)

    # splitting dataset into X and y
    # X = df.iloc[:, 0:-1]
    # y = df.iloc[:, -1]

    # applying PCA function on training and testing set of X components
    # pca = PCA(n_components = 10)
    # X_transformed = pca.fit_transform(X)
    # X_transformed_df = pd.DataFrame(X_transformed, columns=df.columns[:-1])
    
    # reduced_df = pd.DataFrame(X_transformed, columns=df.columns[:-1])
    # reduced_df['target_variable'] = y
    

    scaled_df = pd.DataFrame(StandardScaler().fit_transform(df))
    pca = PCA(n_components = 10)
    pca.fit(scaled_df)
    data_pca = pca.transform(scaled_df)
    pca_df = pd.DataFrame(data_pca,index=df.index[:])

    # export reduced dataframe to csv file for analysis
    pca_df.to_csv('reduced.csv', mode="w")

    print("PCA Data Reduction completed ✓")









#     ########### LDA isn't working atm. probably an issue with formatting of df ##########
#     ########### have removed GEHC from nasdaq_100_tickers.txt as it has blank data. can re-add it with "fix blank data" stuff once lda is workiing

#     # # Separate the features from the target variable
#     # X = df.iloc[1:, 1:-1]
#     # y = df.iloc[1:, -1]
#     # # Create and fit the LDA model
#     # n_components = 10
#     # lda = LinearDiscriminantAnalysis(n_components=n_components)
#     # lda.fit(X, y)
#     # # Transform the DataFrame into a reduced dimension space
#     # X_transformed = lda.transform(X)
#     # # Combine the transformed features with the target variable
#     # reduced_df = pd.DataFrame(X_transformed, columns=df.columns[:-1])
#     # reduced_df['target_variable'] = y

#     # Separate the features from the target variable
#     X = df.iloc[:, :-1].values
#     y = df.iloc[:, -1].values

#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Create a LinearDiscriminantAnalysis object
#     lda = LinearDiscriminantAnalysis()

#     # Fit the LDA model to the training data
#     lda.fit(X_train, y_train)

#     # Transform the training and testing data
#     X_train_transformed = lda.transform(X_train)
#     X_test_transformed = lda.transform(X_test)

#     # Reduce the number of features from 250 to 10
#     X_train_reduced = X_train_transformed[:, :10]
#     X_test_reduced = X_test_transformed[:, :10]

#     # Print the reduced datasets
#     print("Reduced training dataset:")
#     print(X_train_reduced)

#     print("Reduced testing dataset:")
#     print(X_test_reduced)

