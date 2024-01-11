### Package imports ###
# ------------------- #
import pandas as pd
import numpy as np
import matplotlib.figure as figure
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
    arima_predictions = list()
    model = auto_arima(ticker_row, trace=True, error_action="ignore", suppress_warnings=True)  # auto finds the ideal p,d,q values
    model_order = model.get_params()["order"]
    # redict
    model = ARIMA(history, order=model_order)  # defines model
    model_fit = model.fit()  # fits model
    yhat = model_fit.forecast()[0]  # initial forecast
    arima_predictions.append(yhat)
    history.append(y[0])

    # rolling forecasts
    for i in range(1, len(y)):
        # predict
        model = ARIMA(history, order=(model_order))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        arima_predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)

    # report performance
    mse = mean_squared_error(y, arima_predictions)
    print('MSE: '+str(mse))
    mae = mean_absolute_error(y, arima_predictions)
    print('MAE: '+str(mae))
    rmse = math.sqrt(mean_squared_error(y, arima_predictions))
    print('RMSE: '+str(rmse))
    rmspe = np.sqrt(np.mean(np.square(((y - arima_predictions) / y)))) * 100
    print("RMSPE:", rmspe)

    fig = figure.Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.plot(ticker_row, color='green', label='Train Stock Price')
    ax.plot(test.index, y, color='red', label='Real Stock Price')
    ax.plot(test.index, arima_predictions, color='blue', label='Predicted Stock Price')
    ax.legend()
    ax.set_title("Nasdaq Closing Price Forecast (ARIMA)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (USD)")
    
    graph = fig

    return graph

# Facebook Prophet #
def ml_facebook_prophet(ticker, forecast_length):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    df_model = pd.DataFrame(ticker_row)
    df_model = df_model.reset_index()
    df_model.columns = ['ds', 'y']

    fb_model = Prophet()
    fb_model.fit(df_model)

    future = fb_model.make_future_dataframe(periods=forecast_length)
    prophet_predictions = fb_model.predict(future)

    fig = figure.Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    fb_model.plot(prophet_predictions, ax=ax)

    ax.legend()
    ax.set_title("Nasdaq Closing Price Forecast (Prophet)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (USD)")
    
    graph = fig

    return (fb_model, prophet_predictions, graph)

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

    fig = figure.Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.plot(test_y, label="Actual Price")
    ax.plot(predictions, label="Predicted Price")
    ax.legend()
    ax.set_title("Nasdaq Closing Price Forecast (LSTM)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Closing Price (USD)")
    
    graph = fig

    return graph

# Linear Regression # 
def ml_linear_regression(ticker, forecast_length):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    # extract close prices and reshape for linear regression
    close_prices = ticker_row.values.reshape(-1, 1)

    # create linear regression model
    model = LinearRegression()

    # use model to make extended predictions
    model.fit(np.arange(len(close_prices)).reshape(-1, 1), close_prices)
    extend_lin_x = np.arange(0, len(close_prices) + forecast_length)
    extend_lin_y = model.predict(extend_lin_x.reshape(-1, 1))

    # plot actual and predicted values
    fig = figure.Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.plot(close_prices, label="Actual Price")
    ax.plot(extend_lin_x, extend_lin_y, label="Predicted Price")
    ax.legend()
    ax.set_title("Nasdaq Closing Price Forecast (Linear Regression)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Closing Price (USD)")
    
    graph = fig

    return (extend_lin_x, extend_lin_y, graph)


def forecasting(ticker, prophet_predictions, linear_regression):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    prophet_model = prophet_predictions[0]
    prophet_preds = prophet_predictions[1]
    prophet_graph = prophet_predictions[2]

    prophet_pred_vals = prophet_preds['yhat']

    linear_y = linear_regression[1]

    buy_sell = ""
    if prophet_pred_vals.iloc[-1] < linear_y[-1]:
        buy_sell = "Buy"
    elif prophet_pred_vals.iloc[-1] > linear_y[-1]:
        buy_sell = "Sell"
    else:
        buy_sell = "Hold"

    fig = figure.Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    prophet_model.plot(prophet_preds, ax=ax)

    ax.plot(prophet_preds['ds'], linear_y, color="orange",label="Predicted Price")  # linear regression
    ax.text(prophet_preds['ds'].iloc[180], prophet_preds['yhat'].iloc[-1], f"Buy or Sell?\n\n {buy_sell.upper()}", ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax.legend()
    ax.set_title("Forecasting")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (USD)")

    graph = fig

    return graph


