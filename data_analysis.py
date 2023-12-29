### Package imports ###
# ------------------- #
import matplotlib.pyplot as plt
import matplotlib.figure as figure


### Correlation between analysis tickers ###
# ---------------------------------------- #
def data_correlation(df, analysis_tickers, ticker):

    correlated_data = df.corr()
    correlated_data.to_csv('data/correlated.csv', mode="w")

    print("\nData Corerlation:")
    for i in range(len(correlated_data.index)):  # loops each ticker label in correlated df
        if correlated_data.index[i] in analysis_tickers:  # checks if current label is one of the analysis tickers
            # if yes:
            header = correlated_data.index[i]  # ticker label saved
            row = correlated_data.iloc[i, :]  # row captured
            sorted_row = row.sort_values()  # row ordered

            # top/bottom 10 values captured
            bottom_corr = sorted_row[:10]
            top_corr_11 = sorted_row[-11:]  # 11 selected because highest value is the ticker itself
            
            top_corr = top_corr_11.iloc[:-1]  # deletes its own 1.0 corr value 
            top_corr_reverse = top_corr.iloc[::-1]  # reverses the top vals so it is highest to lowest
            
            # retrieves correlations
            if header == ticker[0]:
                return f"\n{header}\n----\nTop 10:\n{top_corr_reverse}\nBottom 10:\n{bottom_corr}"


### Exploratory Data Analysis ###
# ----------------------------- #

# line chart to show closing prices over the year
def eda_line_chart(ticker):
    
    ticker_name = ticker[0]
    ticker_row = ticker[1]

    fig = figure.Figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    ax.set_title(f"{ticker_name} - Line Chart of Closing Values 1 Year")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (USD)")
    ax.plot(ticker_row)
    graph = fig

    return graph


# box plot to show spread of closing prices
def eda_box_plot(ticker):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    fig = figure.Figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    ax.set_title(f"{ticker_name} - Boxplot of Closing Values 1 Year")
    ax.set_xlabel(ticker_name)
    ax.set_ylabel("Closing Price (USD)")
    ax.boxplot(ticker_row)
    graph = fig

    return graph

# histogram to show frequency of closing prices
def eda_histogram(ticker):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    fig = figure.Figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    ax.set_title(f"{ticker_name} - Histogram Showing Frequency of Closing Values 1 Year")
    ax.set_xlabel("Closing Price (USD)")
    ax.set_ylabel("Frequency")
    ax.hist(ticker_row, bins=10)
    graph = fig

    return graph


