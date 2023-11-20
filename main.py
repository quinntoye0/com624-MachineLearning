### Package imports ###
# ------------------- #
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
# Market Data
import yfinance as yf
# Graphing/Visualisation


# tickers list
with open("nasdaq_100_tickers.txt") as tickersFile:
    tickersList = [line.rstrip('\n') for line in tickersFile]

# override yahoo finance api
yf.pdr_override()
# Import the data frame (df) from yahoo finance using the specified stock as the ticker symbol
df = yf.download(tickers=tickersList,period='1y',interval='1d')

print(df)


