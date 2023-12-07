### File imports ###
# ---------------- #
import nasdaq_data_retrieval as nasDR
import data_grouping as dGroup

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


df1 = nasDR.nasdaq_data_retrieval()
reduced_df = dGroup.pca_reduction(df1)
#dGroup.kmeans(reduced_df)

