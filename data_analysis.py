### Package imports ###
# ------------------- #
import pandas as pd
import matplotlib.pyplot as plt


### Correlation between analysis tickers ###
# ---------------------------------------- #
def data_correlation(df, analysis_tickers):

    correlated_data = df.corr()
    correlated_data.to_csv('data/correlated.csv', mode="w")


    print("\nData Corerlation:")
    for x in range(len(correlated_data.index)):  # loops each ticker label in correlated df
        if correlated_data.index[x] in analysis_tickers:  # checks if current label is one of the analysis tickers
            # if yes:
            header = correlated_data.index[x]  # ticker label saved
            row = correlated_data.iloc[x, :]  # row captured
            sorted_row = row.sort_values()  # row ordered

            # top/bottom 10 values captured
            bottom_corr = sorted_row[:10]
            top_corr_11 = sorted_row[-11:]  # 11 selected because highest value is the ticker itself
            
            top_corr = top_corr_11.iloc[:-1]  # deletes its own 1.0 corr value 
            top_corr_reverse = top_corr.iloc[::-1]  # reverses the top vals so it is highest to lowest
            
            # displays correlations
            print(f"\n{header}\n----\nTop 10:\n{top_corr_reverse}\nBottom 10:\n{bottom_corr}")


### Exploratory Data Analysis ###
# ----------------------------- #

def eda_line_chart(ticker):
    
    ticker_name = ticker[0]
    ticker_row = ticker[1]

    plt.title(f"{ticker_name} - Line Chart of Closing Values 1 Year")
    plt.xlabel("Date")
    plt.ylabel("Closing Value")
    plt.plot(ticker_row)
    plt.show()

def eda_box_plot(ticker):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    plt.figure(figsize=(8, 6))
    plt.boxplot(ticker_row)
    plt.title(f"{ticker_name} - Boxplot of Closing Values 1 Year")
    plt.xlabel(ticker_name)
    plt.ylabel("Closing Price (USD)")
    plt.show()

def eda_histogram(ticker):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    plt.figure(figsize=(8, 6))
    plt.hist(ticker_row, bins=10)
    plt.title(f"{ticker_name} - Histogram Showing Frequency of Closing Values 1 Year")
    plt.xlabel(f"{ticker_name} Closing Price")
    plt.ylabel("Frequency")
    plt.show()