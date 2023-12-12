"""

Module for loading and previewing datasets.

Classes:
- DatasetLoader: A class containing methods for loading and previewing datasets.

"""
import pandas as pd


class DatasetLoader:
    """Class to load dataset and preview it"""

    def load_dataset(filename):
        """
        Loads a dataset from the specified filename.

        Parameters:
        filename (str): The path to the dataset file.

        Returns:
        DataFrame: The loaded dataset as a pandas DataFrame.
        """
        return pd.read_csv(filename)

    def preview_data(dataframe, num_rows=5):
        """
        Displays the first few rows of the dataset.

        Parameters:
        dataframe (DataFrame): The pandas DataFrame to preview.
        num_rows (int): The number of rows to display. Default is 5.

        Returns:
        None: This method prints the first few rows of the DataFrame.
        """
        print(dataframe.head(num_rows))
