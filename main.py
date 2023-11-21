### Package imports ###
# ------------------- #
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
# Market Data
import yfinance as yf
# Graphing/Visualisation

### Nasdaq Data Download ###
# ------------------------ #
# retrieve and create tickers list
print("\nImporting top 100 tickers...")
with open("nasdaq_100_tickers.txt") as tickersFile:
    tickersList = [line.rstrip('\n') for line in tickersFile]
print("Tickers imported ✓")

# override yahoo finance api
yf.pdr_override()
# import the data frame (df) from yahoo finance using the tickersList to grab all stocks
print("\nDownloading Nasdaq data...")
df = yf.download(tickers=tickersList,period='1y',interval='1d')['Close']
# transposes df (flips collumns/rows)
dfTransposed = df.T
print("Nasdaq data downloaded ✓")

# ### Exporting to csv ###
# # Save the dataframe to a CSV file
# dfTransposed.to_csv('Nasdaq.csv')

### Dimensionality Reduction using LDA ###
# -------------------------------------- #
print("\nReducing data size for each stock...")

