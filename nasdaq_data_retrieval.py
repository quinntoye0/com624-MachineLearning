### Package imports ###
# ------------------- #
# Market Data
import yfinance as yf

def nasdaq_data_retrieval():

    ### Nasdaq Data Download ###
    # ------------------------ #

    # retrieve and create tickers list
    print("\nImporting top 100 tickers...")
    with open("data/nasdaq_100_tickers.txt") as tickersFile:
        tickersList = [line.rstrip('\n') for line in tickersFile]
    print("Tickers imported ✓")

    # override yahoo finance api
    yf.pdr_override()
    
    # import the data frame (df) from yahoo finance using the tickersList to grab all stocks
    print("\nDownloading Nasdaq data...")
    df = yf.download(tickers=tickersList,period='1y',interval='1d')['Close']
    dfTransposed = df.T  # transposes df (flips collumns/rows)
    print("Nasdaq data downloaded ✓")

    # ### Exporting to csv ###
    # # Save the dataframe to a CSV file
    dfTransposed.to_csv('data/Nasdaq.csv', mode="w")

    return dfTransposed  # X=Dates, y=Stocks