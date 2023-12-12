### Package imports ###
# ------------------- #
# Market Data
import yfinance as yf
import pandas as pd

def nasdaq_data_retrieval():

    ### Nasdaq Data Download ###
    # ------------------------ #

    # retrieve and create tickers list
    print("\nImporting top 100 tickers...")
    with open("data/nasdaq_100_tickers.txt") as tickers_file:
        tickers_list = [line.rstrip('\n') for line in tickers_file]
    print("Tickers imported ✓")

    # override yahoo finance api
    yf.pdr_override()
    
    # import the data frame (df) from yahoo finance using the tickersList to grab all stocks
    print("\nDownloading Nasdaq data...")
    df = yf.download(tickers=tickers_list,period='1y',interval='1d')['Close']
    df = pd.DataFrame(df)
    df_transposed = df.T  # transposes df (flips collumns/rows)
    print("Nasdaq data downloaded ✓")

    # ### Exporting to csv ###
    # # Save the dataframe to a CSV file
    df_transposed.to_csv('data/Nasdaq.csv', mode="w")

    return (df, df_transposed)  
    # df = X=Stocks, y=Dates 
    # dfTransposed = X=Dates, y=Stocks