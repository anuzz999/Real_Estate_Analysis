# # scraping_data_cleaning.py
"""

This module defines classes for inspecting and cleaning data.

"""


import pandas as pd


class DataInspector:
    """
    Class for inspecting data.

    """

    def get_data_info(dataframe):
        """
        Retrieves information about missing values and data types in the DataFrame.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.

        Returns
        -------
        DataFrame
            DataFrame containing columns for missing values and data types.
        """
        missing_values = dataframe.isnull().sum()
        data_types = dataframe.dtypes
        return pd.DataFrame({"Missing Values": missing_values, "Data Type": data_types})


class DataCleaner:
    """
    Class for cleaning data.

    Methods
    -------
    drop_high_missing_columns(dataframe, threshold_ratio)
        Drops columns with missing values exceeding a specified threshold ratio.

    fill_missing_values(dataframe)
        Fills missing values in the DataFrame using the mode for object columns and the median for numeric columns.

    correct_price_columns(dataframe, price_columns)
        Corrects price columns by removing non-numeric characters and converting to float.
    """

    def drop_high_missing_columns(dataframe, threshold_ratio):
        """
        Drops columns with missing values exceeding a specified threshold ratio.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        threshold_ratio : float
            Threshold ratio for dropping columns.

        Returns
        -------
        DataFrame
            DataFrame with high-missing columns dropped.
        """
        threshold = threshold_ratio * len(dataframe)
        return dataframe.dropna(thresh=threshold, axis=1)

    def fill_missing_values(dataframe):
        """
        Fills missing values in the DataFrame using the mode for object columns and the median for numeric columns.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with missing values filled.
        """
        for column in dataframe.columns:
            if dataframe[column].dtype == "object":
                dataframe[column].fillna(dataframe[column].mode()[0], inplace=True)
            else:
                dataframe[column].fillna(dataframe[column].median(), inplace=True)
        return dataframe

    def correct_price_columns(dataframe, price_columns):
        """
        Corrects price columns by removing non-numeric characters and converting to float.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        price_columns : list
            List of columns representing prices.

        Returns
        -------
        DataFrame
            DataFrame with corrected price columns.
        """
        for col in price_columns:
            if col in dataframe.columns:
                dataframe[col] = (
                    dataframe[col].replace("[^\d.]", "", regex=True).astype(float)
                )
        return dataframe
