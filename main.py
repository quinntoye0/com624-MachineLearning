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
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

### Chosen tickers list for analysis ###
### -------------------------------- ###

analysis_tickers = ['CPRT', 'ORLY', 'BKNG', 'NFLX']

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
cprt = ("CPRT", ana_rows[0])
orly = ("ORLY", ana_rows[1])
bkng= ("BKNG", ana_rows[2])
nflx = ("NFLX", ana_rows[3])
ana_rows = [cprt, orly, bkng, nflx]


### Tkinter GUI ###
# --------------- #
window = tk.Tk()  # initialises gui window
window.state("zoomed")  # makes window fullscreen
window.title("Nasdaq Stock Analysis and Predictions")  # sets window title


### Data Analysis ###
# ----------------- #

graphs = []

# Exploratory Data Analysis #
graphs.append(dAna.eda_line_chart(cprt))  # Line chart with all analysis stocks    ######### REPLACE ROW WITH USER CHOSEN ONE - NOT HARD CODED TICKER ROW
graphs.append(dAna.eda_box_plot(cprt))  # Box plot with all analysis stocks   ######### REPLACE ROW WITH USER CHOSEN ONE - NOT HARD CODED TICKER ROW
graphs.append(dAna.eda_histogram(cprt))  # Histogram with analysis stocks

# Correlation #
correlation = dAna.data_correlation(df, analysis_tickers, cprt)  # Data Correlation 


### Data Predictions ###
# -------------------- #
graphs.append(dPred.ml_arima(cprt))  # ARIMA prediction model
prophet_predictions = dPred.ml_facebook_prophet(cprt, forecast_length=5)  # Facebook Prophet prediction model (forecast_length = 5 (1 week)/10 (2 weeks)/25 (1 month))
graphs.append(prophet_predictions[2])
graphs.append(dPred.ml_lstm(cprt))  # LSTM Prediction Model
linear_regression = dPred.ml_linear_regression(cprt, forecast_length=5)  # Linear Regression Model
graphs.append(linear_regression[2])

graphs.append(dPred.forecasting(cprt, prophet_predictions,  linear_regression))


# Tkinter Window Management #
# ------------------------- #

# set up main frame
main_frame = Frame(window)
main_frame.pack(fill=BOTH, expand=1)

# create canvas
canvas = Canvas(main_frame)
canvas.pack(side=LEFT, fill=BOTH, expand=1)

# add scrollbar to canvas
scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)

# configure canvas
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion = canvas.bbox("all")))

# create second frame to hold window contents
content_frame = Frame(canvas)

# add content_fram to window in canvas 
canvas.create_window((0,0), window=content_frame, anchor="nw")

# display graphs
canvases = []
for i in range(len(graphs)):
    
    fig_canvas = FigureCanvasTkAgg(graphs[i], master=content_frame)
    canvases.append(fig_canvas)

for i in range(4):  # if i > 0 then for j. else add the top bar with dropmenus
    for j in range(2):
        canvases[i*2 + j].get_tk_widget().grid(row=i, column=j)

correlation_label = tk.Label(content_frame, text=correlation).grid(row=5, column=0)


# plot_frame = tk.Frame(window)
# plot_frame.pack()

# canvas1 = FigureCanvasTkAgg(eda_line_chart, master=window)
# canvas1.get_tk_widget().grid(row=0, column=0)
# canvas1.draw()
# #canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# canvas2 = FigureCanvasTkAgg(eda_box_plot, master=window)
# canvas2.get_tk_widget().grid(row=0, column=1)
# canvas2.draw()

window.mainloop()

