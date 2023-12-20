### Package imports ###
# ------------------- #
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from prophet import Prophet


### Correlation between analysis tickers ###
# ---------------------------------------- #
def data_correlation(df, analysis_tickers):

    correlated_data = df.corr()
    correlated_data.to_csv('data/correlated.csv', mode="w")


    print("\nData Corerlation:")
    for x in range(len(correlated_data.index)):  # loops each ticker label in correlated df
        if correlated_data.index[x] in analysis_tickers:  # checks if current label is one of the analysis tickers
            # if yes:
            header = correlated_data.index[x]  # ticker label saved
            row = correlated_data.iloc[x, :]  # row captured
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

    start_date = ticker_row.index[-1].date()

    model = auto_arima(ticker_row, trace=True, error_action="ignore", suppress_warnings=True)
    model_order = model.get_params()["order"]

    model = ARIMA(ticker_row, order=model_order)
    model_fit = model.fit()

    n_periods = 30  # number of forecast periods
    forecast, stderr, conf_int = model_fit.forecast(ticker_row.size, alpha=0.5)
    forecast_series = pd.Series(forecast, index=ticker_row.index)
    lower_series = pd.Series(conf_int[:, 0], index=ticker_row.index)
    upper_series = pd.Series(conf_int[:, 0], index=ticker_row.index)

    plt.plot(ticker_row, colour = 'blue', label='Actual Stock Price')
    plt.plot(forecast_series, colour = 'red', label='Predicted Stock Price')
    plt.fill_between(lower_series.index, lower_series, upper_series, color="lightgray", alpha=0.5)
    
    # plt.plot(ticker_row.index, ticker_row, label="Actual")
    # plt.plot(forecast.index, forecast, label="Forecast")
    # plt.fill_between(forecast.index, conf_int[:, 0], conf_int[:, 1], color="lightgray", alpha=0.5)

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Actual Stock Price")
    plt.title("ARIMA Model Forecast")
    plt.show()


def ml_facebook_prophet(ticker):

    ticker_name = ticker[0]
    ticker_row = ticker[1]

    print(ticker_row)

    df_model = pd.DataFrame(ticker_row)
    df_model = df_model.reset_index()
    df_model.columns = ['ds', 'y']
    
    print(df_model)

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


