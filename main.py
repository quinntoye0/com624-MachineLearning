### File imports ###
# ---------------- #
import nasdaqDataRetrieval as nasDR
import dataGrouping as dGroup

### Package imports ###
# ------------------- #
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import tkinter as tk



# ### Tkinter Test ###
# # ---------------- #
# window = tk.Tk()
# helloWorld = tk.Label(text="Hello WWorld")
# helloWorld.pack()
# window.mainloop()


df1 = nasDR.nasdaqDataRetrieval()
df2 = dGroup.dataGrouping(df1)

