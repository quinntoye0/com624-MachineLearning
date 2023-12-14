### File imports ###
# ---------------- #
import nasdaq_data_retrieval as nasDR
import data_grouping as dGroup
import data_analysis as dAna

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

ana_rows = dGroup.select_analysis_rows(df_transposed, analysis_tickers)  # select specific rows from df for analysis
# splitting rows
amd = ("AMD", ana_rows[0])
orly = ("ORLY", ana_rows[1])
bkng= ("BKNG", ana_rows[2])
nflx = ("NFLX", ana_rows[3])

'''reduced_df = dGroup.pca_reduction(df_transposed) '''  # PCA Reduction
'''dGroup.kmeans(reduced_df) '''  # KMeans Clustering

### Data Analysis ###
# ----------------- #

dAna.data_correlation(df, analysis_tickers)  # Data Correlation #

# Exploratory Data Analysis #
dAna.display_shape(df, amd)  ######### REPLACE ROW WITH USER CHOSEN ONE - NOT HARD CODED TICKER ROW

