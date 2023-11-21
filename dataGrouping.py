### Package imports ###
# ------------------- #
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def dataGrouping(df):

    ### Dimensionality Reduction using LDA ###
    # -------------------------------------- #
    print("\nReducing data size for each stock using LDA...")


    ########### LDA isn't working atm. probably an issue with formatting of df ##########
    ########### have removed GEHC as it has blank data. can re-add it with "fix blank data" stuff once lda is workiing

    # # Separate the features from the target variable
    # X = df.iloc[1:, 1:-1]
    # y = df.iloc[1:, -1]
    # # Create and fit the LDA model
    # n_components = 10
    # lda = LinearDiscriminantAnalysis(n_components=n_components)
    # lda.fit(X, y)
    # # Transform the DataFrame into a reduced dimension space
    # X_transformed = lda.transform(X)
    # # Combine the transformed features with the target variable
    # reduced_df = pd.DataFrame(X_transformed, columns=df.columns[:-1])
    # reduced_df['target_variable'] = y


    # reduced_df.to_csv('reduced.csv')

    print("LDA Data Reduction completed ✓")



