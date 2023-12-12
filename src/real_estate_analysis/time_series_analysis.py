# time_series_analysis.py

"""

This module provides a TimeSeriesModeler class for ARIMA time series modeling.

"""

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import warnings


class TimeSeriesModeler:
    """
    Class for ARIMA time series modeling.

    """

    def __init__(self):
        # Constructor can be used for setting common properties
        pass

    def prepare_data(self, dataframe, region_name, value_column, date_column):
        """
        Prepares time series data by filtering and setting the index.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame containing time series data.
        region_name : str
            Name of the column representing regions.
        value_column : str
            Name of the column representing the values to be modeled.
        date_column : str
            Name of the column representing the date.

        Returns
        -------
        DataFrame
            Time series data prepared for modeling.
        """
        filtered_data = dataframe[dataframe[region_name] == "United States"]
        ts_data = filtered_data[[date_column, value_column]].dropna()
        ts_data.set_index(date_column, inplace=True)
        return ts_data

    def split_data(self, ts_data, train_end_date, test_start_date):
        """
        Splits time series data into training and testing sets.

        Parameters
        ----------
        ts_data : DataFrame
            Time series data to be split.
        train_end_date : str
            End date for the training data.
        test_start_date : str
            Start date for the testing data.

        Returns
        -------
        DataFrame
            Training data.
        DataFrame
            Testing data.
        """
        train_data = ts_data[:train_end_date]
        test_data = ts_data[test_start_date:]
        return train_data, test_data

    def fit_arima_model(self, train_data, arima_order):
        """
        Fits an ARIMA model to the training data.

        Parameters
        ----------
        train_data : DataFrame
            Training data for model fitting.
        arima_order : tuple
            Order of the ARIMA model (p, d, q).

        Returns
        -------
        ARIMAResultsWrapper
            Fitted ARIMA model results.
        """
        warnings.filterwarnings("ignore")
        arima_model = ARIMA(train_data, order=arima_order)
        arima_result = arima_model.fit()
        return arima_result

    def display_model_summary(self, arima_result):
        """
        Displays the summary of the fitted ARIMA model.

        Parameters
        ----------
        arima_result : ARIMAResultsWrapper
            Fitted ARIMA model results.

        Returns
        -------
        str
            Model summary as a string.
        """
        return arima_result.summary()

    def forecast_arima(self, model_fit, steps):
        """
        Forecasts future values using the fitted ARIMA model.

        Parameters
        ----------
        model_fit : ARIMAResultsWrapper
            Fitted ARIMA model results.
        steps : int
            Number of steps to forecast into the future.

        Returns
        -------
        Series
            Forecasted values.
        """
        forecast = model_fit.forecast(steps=steps)
        return forecast

    def plot_forecast(self, train_data, test_data, forecast, title, x_label, y_label):
        """
        Plots the training data, actual prices, and predicted prices.

        Parameters
        ----------
        train_data : DataFrame
            Training data.
        test_data : DataFrame
            Testing data.
        forecast : Series
            Forecasted values.
        title : str
            Title of the plot.
        x_label : str
            Label for the x-axis.
        y_label : str
            Label for the y-axis.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(train_data, label="Training Data")
        plt.plot(test_data, label="Actual Prices", color="orange")
        plt.plot(test_data.index, forecast, label="Predicted Prices", color="green")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    def forecast_with_confidence_interval(self, model_fit, steps):
        """
        Forecasts future values with confidence intervals using the fitted ARIMA model.

        Parameters
        ----------
        model_fit : ARIMAResultsWrapper
            Fitted ARIMA model results.
        steps : int
            Number of steps to forecast into the future.

        Returns
        -------
        Series
            Forecasted values.
        DataFrame
            Confidence intervals.
        """
        forecast = model_fit.get_forecast(steps=steps)
        forecast_values = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()
        return forecast_values, confidence_intervals

    def plot_forecast_with_intervals(
        self,
        train_data,
        test_data,
        forecast_values,
        confidence_intervals,
        title,
        x_label,
        y_label,
    ):
        """
        Plots the training data, actual data, forecasted values, and confidence intervals.

        Parameters
        ----------
        train_data : DataFrame
            Training data.
        test_data : DataFrame
            Testing data.
        forecast_values : Series
            Forecasted values.
        confidence_intervals : DataFrame
            Confidence intervals.
        title : str
            Title of the plot.
        x_label : str
            Label for the x-axis.
        y_label : str
            Label for the y-axis.
        """
        plt.figure(figsize=(15, 8))
        plt.plot(train_data, label="Training Data", color="blue")
        plt.plot(test_data, label="Actual Data", color="green")
        plt.plot(
            forecast_values, label="Forecasted Values", color="red", linestyle="--"
        )
        plt.fill_between(
            confidence_intervals.index,
            confidence_intervals.iloc[:, 0],
            confidence_intervals.iloc[:, 1],
            color="pink",
            alpha=0.3,
        )
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        plt.show()
