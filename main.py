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

analysisTickers = ['AMD', 'ORLY', 'BKNG', 'NFLX']

# ### Tkinter Test ###
# # ---------------- #
# window = tk.Tk()
# helloWorld = tk.Label(text="Hello WWorld")
# helloWorld.pack()
# window.mainloop()


dfTuple = nasDR.nasdaq_data_retrieval()
df = dfTuple[0]  # original nasdaq data download
df_transposed = dfTuple[1]  # nasdaq data with axis transposed

reduced_df = dGroup.pca_reduction(df_transposed)
#dGroup.kmeans(reduced_df)
dAna.data_correlation(df)