### Package imports ###
# ------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import Dense,LSTM
from keras.models import Sequential
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
# hides TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


### Correlation between analysis tickers ###
# ---------------------------------------- #
def data_correlation(df, analysis_tickers):

    correlated_data = df.corr()
    correlated_data.to_csv('data/correlated.csv', mode="w")


    print("\nData Corerlation:")
    for i in range(len(correlated_data.index)):  # loops each ticker label in correlated df
        if correlated_data.index[i] in analysis_tickers:  # checks if current label is one of the analysis tickers
            # if yes:
            header = correlated_data.index[i]  # ticker label saved
            row = correlated_data.iloc[i, :]  # row captured
            sorted_row = row.sort_values()  # row ordered

            # top/bottom 10 values captured
            bottom_corr = sorted_row[:10]
            top_corr_11 = sorted_row[-11:]  # 11 selected because highest value is the ticker itself
            
            top_corr = top_corr_11.iloc[:-1]  # deletes its own 1.0 corr value 
            top_corr_reverse = top_corr.iloc[::-1]  # reverses the top vals so it is highest to lowest
            
            # displays correlations
            print(f"\n{header}\n----\nTop 10:\n{top_corr_reverse}\nBottom 10:\n{bottom_corr}")


### Exploratory Data Analysis ###
# ----------------------------- #

# line chart to show closing prices over the year
def eda_line_chart(ticker):
    
    ticker_name = ticker[0]
    ticker_row = ticker[1]

    plt.title(f"{ticker_name} - Line Chart of Closing Values 1 Year")
    plt.xlabel("Date")
    plt.ylabel("Closing Value")
    plt.plot(ticker_row)
    plt.show()

# box plot to show spread of closing prices
def eda_box_plot(ticker):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    plt.figure(figsize=(8, 6))
    plt.boxplot(ticker_row)
    plt.title(f"{ticker_name} - Boxplot of Closing Values 1 Year")
    plt.xlabel(ticker_name)
    plt.ylabel("Closing Price (USD)")
    plt.show()

# histogram to show frequency of closing prices
def eda_histogram(ticker):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    plt.figure(figsize=(8, 6))
    plt.hist(ticker_row, bins=10)
    plt.title(f"{ticker_name} - Histogram Showing Frequency of Closing Values 1 Year")
    plt.xlabel(f"{ticker_name} Closing Price")
    plt.ylabel("Frequency")
    plt.show()


### Machine Learning Models ###
# --------------------------- #

# ARIMA #
def ml_arima(ticker):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    train_size = int(len(ticker_row)*0.75)
    train, test = ticker_row[:train_size], ticker_row[train_size:]

    history = [x for x in train]
    y = test

    # make first prediction
    predictions = list()
    model = auto_arima(ticker_row, trace=True, error_action="ignore", suppress_warnings=True)  # auto finds the ideal p,d,q values
    model_order = model.get_params()["order"]
    # redict
    model = ARIMA(history, order=model_order)  # defines model
    model_fit = model.fit()  # fits model
    yhat = model_fit.forecast()[0]  # initial forecast
    predictions.append(yhat)
    history.append(y[0])

    # rolling forecasts
    for i in range(1, len(y)):
        # predict
        model = ARIMA(history, order=(1,1,0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)

    # report performance
    mse = mean_squared_error(y, predictions)
    print('MSE: '+str(mse))
    mae = mean_absolute_error(y, predictions)
    print('MAE: '+str(mae))
    rmse = math.sqrt(mean_squared_error(y, predictions))
    print('RMSE: '+str(rmse))

    plt.figure(figsize=(16,8))
    plt.plot(ticker_row, color='green', label = 'Train Stock Price')
    plt.plot(test.index, y, color = 'red', label = 'Real Stock Price')
    plt.plot(test.index, predictions, color = 'blue', label = 'Predicted Stock Price')

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"{ticker_name} - Nasdaq Closing Price Forecast (ARIMA)")
    plt.show()

# Facebook Prophet #
def ml_facebook_prophet(ticker):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    df_model = pd.DataFrame(ticker_row)
    df_model = df_model.reset_index()
    df_model.columns = ['ds', 'y']

    fb_model = Prophet()
    fb_model.fit(df_model)

    future = fb_model.make_future_dataframe(periods=100)
    forecast = fb_model.predict(future)

    prophet_graph = fb_model.plot(forecast)
    prophet_graph.show()

    plt.xlabel("Date")
    plt.ylabel(f"Nasdaq Closing Price")
    plt.title(f"{ticker_name} - Nasdaq Closing Price Forecast (Prophet)")
    plt.show()

# LSTM #
def ml_lstm(ticker):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    scaler = MinMaxScaler()
    ticker_row2 = scaler.fit_transform(np.array(ticker_row).reshape(-1,1))

    train_size = int(len(ticker_row2)*0.65)
    train, test = ticker_row2[:train_size], ticker_row2[train_size:]    

    def create_dataset(X, time_steps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X[i:i + time_steps]
            Xs.append(v)
            ys.append(X[i + time_steps])
        return np.array(Xs), np.array(ys)

    time_steps = 10
    train_X, train_y = create_dataset(train, time_steps)
    test_X, test_y = create_dataset(test, time_steps)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001))

    # train the model
    model.fit(train_X, train_y, epochs=30, batch_size=16, validation_split=0.1, verbose=1, shuffle=False)

    # make predictions
    predictions = model.predict(test_X)
    predictions = predictions.reshape(78, 10)  # reshapes to 2D

    # plot actual and predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(test_y, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.title(f"{ticker_name} - LSTM")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


# Linear Regression # 
def ml_linear_regression(ticker):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    # extract close prices and reshape for linear regression
    close_prices = ticker_row.values.reshape(-1, 1)

    # create linear regression model
    model = LinearRegression()
    model.fit(np.arange(len(close_prices)).reshape(-1, 1), close_prices)

    # use model to predict values
    predictions = model.predict(np.arange(len(close_prices)).reshape(-1, 1))

    # plot actual and predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(close_prices, label="Actual Price")
    plt.plot(np.arange(len(close_prices)), predictions, label="Predicted Price")
    plt.title(f"{ticker_name} - Linear Regression")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()


