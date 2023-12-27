### File imports ###
# ---------------- #
import nasdaq_data_retrieval as nasDR
import data_grouping as dGroup
import data_analysis as dAna
import data_predictions as dPred

### Package imports ###
# ------------------- #
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import tkinter as tk

### Chosen tickers list for analysis ###
### -------------------------------- ###

analysis_tickers = ['AMD', 'ORLY', 'BKNG', 'NFLX']

# ### Tkinter Test ###
# # ---------------- #
# window = tk.Tk()
# helloWorld = tk.Label(text="Hello WWorld")
# helloWorld.pack()
# window.mainloop()

### YFinance Nasdaq Data Retrieval ###
# ---------------------------------- #
dfTuple = nasDR.nasdaq_data_retrieval()  # retrieve fulll nasdaq data from yfinance
df = dfTuple[0]  # original nasdaq data download
df_transposed = dfTuple[1]  # nasdaq data with axis transposed

### Data Grouping ###
# ----------------- #

'''reduced_df = dGroup.pca_reduction(df_transposed) '''  # PCA Reduction
'''dGroup.kmeans(reduced_df) '''  # KMeans Clustering

ana_rows = dGroup.select_analysis_rows(df_transposed, analysis_tickers)  # select specific rows from df for analysis
# splitting rows
amd = ("AMD", ana_rows[0])
orly = ("ORLY", ana_rows[1])
bkng= ("BKNG", ana_rows[2])
nflx = ("NFLX", ana_rows[3])
ana_rows = [amd, orly, bkng, nflx]


### Data Analysis ###
# ----------------- #

'''dAna.data_correlation(df, anal ysis_tickers) ''' # Data Correlation #

# Exploratory Data Analysis #
'''dAna.eda_line_chart(amd) ''' ######### REPLACE ROW WITH USER CHOSEN ONE - NOT HARD CODED TICKER ROW
'''dAna.eda_box_plot(amd) ''' # Box plot with all analysis stocks   ######### REPLACE ROW WITH USER CHOSEN ONE - NOT HARD CODED TICKER ROW
'''dAna.eda_histogram(amd) ''' # Histogram with analysis stocks


### Data Predictions ###
# -------------------- #
'''dPred.ml_arima(amd) ''' # ARIMA prediction model
dPred.ml_facebook_prophet(bkng)  # Facebook Prophet prediction model
'''dPred.ml_lstm(bkng) ''' # LSTM Prediction Model
'''dPred.ml_linear_regression(amd) ''' # Linear Regression Model