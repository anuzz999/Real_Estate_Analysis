# data_cleaning.py

"""
This module provides a set of functions for cleaning and manipulating pandas DataFrames.

"""
import pandas as pd


class DataCleaner:
    """DataCleaner class with functions to clean and manipulate the DataFrame"""

    def forward_fill(dataframe, columns):
        """
        Forward-fill missing values in specific columns of a DataFrame.

        Parameters:
            dataframe (pd.DataFrame): The input DataFrame.
            columns (str or list): The column or list of columns to forward-fill.

        Returns:
            pd.DataFrame: DataFrame with missing values in specified columns forward-filled.
        """
        df_cleaned = dataframe.copy()
        df_cleaned[columns] = df_cleaned[columns].fillna(method="ffill")
        return df_cleaned

    def drop_columns(dataframe, columns):
        """
        Drop specified columns from a DataFrame.

        Parameters:
            dataframe (pd.DataFrame): The input DataFrame.
            columns (str or list): The column or list of columns to drop.

        Returns:
            pd.DataFrame: DataFrame with specified columns dropped.
        """
        return dataframe.drop(columns=columns)

    def convert_to_datetime(dataframe, column_name, date_format="%Y-%m-%d"):
        """
        Convert a specific column to datetime format in a DataFrame.

        Parameters:
            dataframe (pd.DataFrame): The input DataFrame.
            column_name (str): The column to convert.
            date_format (str, optional): Format of the date. Default is "%Y-%m-%d".

        Returns:
            pd.DataFrame: DataFrame with the specified column converted to datetime format.
        """
        df_converted = dataframe.copy()
        df_converted[column_name] = pd.to_datetime(
            df_converted[column_name], format=date_format
        )
        return df_converted

    def forwards_fill(dataframe, axis=1):
        """
        Forward-fill missing values in a DataFrame along a specified axis.

        Parameters:
            dataframe (pd.DataFrame): The input DataFrame.
            axis (int, optional): The axis along which to forward-fill. Default is 1 (columns).

        Returns:
            pd.DataFrame: DataFrame with missing values forward-filled along the specified axis.
        """
        return dataframe.fillna(method="ffill", axis=axis)

    def reshape_data_long(dataframe, id_vars, var_name, value_name):
        """
        Reshape a DataFrame to long format using pd.melt.

        Parameters:
            dataframe (pd.DataFrame): The input DataFrame.
            id_vars (str or list): Columns to use as identifier variables.
            var_name (str): Name to use for the 'variable' column.
            value_name (str): Name to use for the 'value' column.

        Returns:
            pd.DataFrame: Reshaped DataFrame in long format.
        """
        return pd.melt(
            dataframe, id_vars=id_vars, var_name=var_name, value_name=value_name
        )

    def convert_column_to_datetime(dataframe, column_name):
        """
        This function converts a specific column in a pandas DataFrame to datetime format.

        Parameters:
            dataframe: The pandas DataFrame containing the column to be converted.
            column_name: The name of the column to be converted to datetime.

        Returns:
            A new pandas DataFrame with the specified column converted to datetime format.
        """
        df_converted = dataframe.copy()
        df_converted[column_name] = pd.to_datetime(df_converted[column_name])
        return df_converted

    def get_data_types(dataframe):
        """
        This function returns the data types of all columns in a pandas DataFrame.

        Parameters:
            dataframe: The pandas DataFrame whose data types are to be retrieved.

        Returns:
            A Series containing the data types of all columns in the DataFrame.
        """
        return dataframe.dtypes

    def identify_non_numeric_entries(dataframe, column_name):
        """
        This function identifies and returns rows in a pandas DataFrame where a specific column contains non-numeric entries.

        Parameters:
            dataframe: The pandas DataFrame to be analyzed for non-numeric entries.
            column_name: The name of the column to be examined.

        Returns:
            A new pandas DataFrame containing only rows where the specified column contains non-numeric values.
        """
        return dataframe[pd.to_numeric(dataframe[column_name], errors="coerce").isna()]

    def convert_column_to_numeric_and_fillna(dataframe, column_name, method="ffill"):
        """
        This function converts a specific column in a pandas DataFrame to numeric format and fills in missing values using the specified method.

        Parameters:
            dataframe: The pandas DataFrame containing the column to be converted.
            column_name: The name of the column to be converted to numeric and filled with missing values.
            method: The method to be used for filling missing values. Default option is "ffill" (forward fill).

        Returns:
            A new pandas DataFrame with the specified column converted to numeric format and missing values filled.
        """
        df_converted = dataframe.copy()
        df_converted[column_name] = pd.to_numeric(
            df_converted[column_name], errors="coerce"
        )
        df_converted[column_name] = df_converted[column_name].fillna(method=method)
        return df_converted
