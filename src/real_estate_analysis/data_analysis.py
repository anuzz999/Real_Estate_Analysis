# data_analysis.py

"""

Module for analyzing and visualizing time-series data.


"""

import matplotlib.pyplot as plt
import pandas as pd


class DataAnalyzer:
    """Class DataAnalyzer to clean, identify missing_values, and preprocess and analyze the data"""

    def check_missing_values(dataframe):
        """Check and return the count of missing values in the DataFrame."""
        return dataframe.isnull().sum()

    def get_statistical_summary(dataframe):
        """Return the statistical summary of the DataFrame."""
        return dataframe.describe()

    def identify_missing_value_columns(dataframe):
        """Identify and return the columns with missing values."""
        return dataframe.columns[dataframe.isna().any()].tolist()

    def display_initial_data(dataframe, num_rows=5):
        """Display the first few rows of the DataFrame."""
        return dataframe.head(num_rows)

    def filter_data_for_plotting(
        dataframe, region_name, columns_to_drop, time_series_column="Inventory"
    ):
        """
        Filter and reshape data for plotting time-series.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.
        - region_name (str): The name of the region to filter.
        - columns_to_drop (list): Columns to drop from the DataFrame.
        - time_series_column (str): Name of the time-series column.

        Returns:
        pd.DataFrame: Reshaped time-series data.
        """
        filtered_data = dataframe[dataframe["RegionName"] == region_name]
        time_series_data = filtered_data.drop(columns=columns_to_drop)
        time_series_data = time_series_data.T
        time_series_data.columns = [time_series_column]
        time_series_data.index = pd.to_datetime(time_series_data.index)
        return time_series_data

    def plot_time_series(
        data, title, x_label, y_label, figsize=(15, 6), marker="o", markersize=2
    ):
        """
        Plot time-series data with specified title and labels.

        Parameters:
        - data (pd.Series): Time-series data to plot.
        - title (str): Title of the plot.
        - x_label (str): Label for the x-axis.
        - y_label (str): Label for the y-axis.
        - figsize (tuple): Figure size.
        - marker (str): Marker style.
        - markersize (int): Marker size.

        Returns:
        None: Displays the plot.
        """
        plt.figure(figsize=figsize)
        plt.plot(data, marker=marker, markersize=markersize)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.show()

    def add_year_column(dataframe, date_column):
        """
        Add a new column 'Year' based on the 'date_column'.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.
        - date_column (str): The name of the date column.

        Returns:
        pd.DataFrame: DataFrame with an additional 'Year' column.
        """
        df_with_year = dataframe.copy()
        df_with_year["Year"] = df_with_year[date_column].dt.year
        return df_with_year

    def calculate_annual_mean_inventory(dataframe, group_by_columns, value_column):
        """
        Calculate the annual mean of inventory based on specified grouping columns.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.
        - group_by_columns (list): Columns to group by.
        - value_column (str): Column for which to calculate the mean.

        Returns:
        pd.DataFrame: DataFrame with annual mean inventory.
        """
        return dataframe.groupby(group_by_columns)[value_column].mean().reset_index()

    def calculate_yoy_change(dataframe, group_column, value_column):
        """
        Calculate year-over-year percentage change for the specified group and value columns.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.
        - group_column (str): Column for grouping.
        - value_column (str): Column for which to calculate the percentage change.

        Returns:
        pd.DataFrame: DataFrame with year-over-year percentage change.
        """
        df_with_yoy = dataframe.copy()
        df_with_yoy["YoY_Change"] = (
            df_with_yoy.groupby(group_column)[value_column].pct_change() * 100
        )
        return df_with_yoy

    def calculate_annual_sum(dataframe, group_by_columns, value_column):
        """
        Calculate the annual sum of a specified value column based on grouping columns.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.
        - group_by_columns (list): Columns to group by.
        - value_column (str): Column for which to calculate the sum.

        Returns:
        pd.DataFrame: DataFrame with annual sum values.
        """
        return dataframe.groupby(group_by_columns)[value_column].sum().reset_index()

    def calculate_correlation(dataframe, column1, column2):
        """
        Calculate the correlation between two columns in the DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.
        - column1 (str): Name of the first column.
        - column2 (str): Name of the second column.

        Returns:
        float: Correlation coefficient between the two columns.
        """
        return dataframe[[column1, column2]].corr().iloc[0, 1]

    def calculate_yearly_change(dataframe, date_column, value_column):
        """
        Calculate the yearly percentage change for the specified date and value columns.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.
        - date_column (str): Column containing the date information.
        - value_column (str): Column for which to calculate the change.

        Returns:
        pd.Series: Series with yearly percentage changes.
        """
        dataframe["Year"] = dataframe[date_column].dt.year
        yearly_data = dataframe.groupby("Year")[value_column].mean()
        return yearly_data.pct_change() * 100

    def analyze_seasonal_trends(dataframe, date_column, value_columns):
        """
        Analyze and return seasonal trends in the DataFrame based on specified date and value columns.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.
        - date_column (str): Column containing the date information.
        - value_columns (list): Columns for which to analyze seasonal trends.

        Returns:
        dict: Dictionary containing monthly trends for each specified value column.
        """
        dataframe["Month-Year"] = dataframe[date_column].dt.to_period("M")
        monthly_data = {}
        for value_column in value_columns:
            monthly_data[value_column] = dataframe.groupby("Month-Year")[
                value_column
            ].mean()
        return monthly_data

    def filter_data_for_period(dataframe, date_column, start_year, end_year):
        """
        Filter and return data for a specified period based on start and end years.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.
        - date_column (str): Column containing the date information.
        - start_year (int): Start year of the period.
        - end_year (int): End year of the period.

        Returns:
        pd.DataFrame: DataFrame containing data for the specified period.
        """
        return dataframe[
            (dataframe[date_column].dt.year >= start_year)
            & (dataframe[date_column].dt.year <= end_year)
        ]

    def calculates_yearly_changes(dataframe, date_column, value_columns):
        """
        Calculate yearly percentage changes for specified date and value columns.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.
        - date_column (str): Column containing the date information.
        - value_columns (list): Columns for which to calculate yearly changes.

        Returns:
        dict: Dictionary containing yearly changes for each specified value column.
        """
        dataframe["Year"] = dataframe[date_column].dt.year
        yearly_changes = {}
        for value_column in value_columns:
            yearly_data = dataframe.groupby("Year")[value_column].mean()
            yearly_changes[value_column] = yearly_data.pct_change() * 100
        return yearly_changes

    def filter_data_for_pandemic_analysis(
        self, dataframe, date_column, start_year, end_year
    ):
        """
        Filter and return data for a specified period during a pandemic based on start and end years.

        Parameters:
        - self (DataAnalyzer): The instance of the DataAnalyzer class.
        - dataframe (pd.DataFrame): The input DataFrame.
        - date_column (str): Column containing the date information.
        - start_year (int): Start year of the pandemic period.
        - end_year (int): End year of the pandemic period.

        Returns:
        pd.DataFrame: DataFrame containing data for the specified pandemic period.
        """
        filtered_data = dataframe[
            (dataframe[date_column].dt.year >= start_year)
            & (dataframe[date_column].dt.year <= end_year)
        ]
        return filtered_data

    def analyze_monthly_trends(self, dataframe, date_column, value_columns):
        """
        Analyze and return monthly trends in the DataFrame based on specified date and value columns.

        Parameters:
        - self (DataAnalyzer): The instance of the DataAnalyzer class.
        - dataframe (pd.DataFrame): The input DataFrame.
        - date_column (str): Column containing the date information.
        - value_columns (list): Columns for which to analyze monthly trends.

        Returns:
        dict: Dictionary containing monthly trends for each specified value column.
        """
        dataframe["Month-Year"] = dataframe[date_column].dt.to_period("M")
        monthly_trends = {}
        for value_column in value_columns:
            monthly_data = dataframe.groupby("Month-Year")[value_column].mean()
            monthly_trends[value_column] = monthly_data
        return monthly_trends
