"""

Module for loading various real estate datasets and performing data manipulation.


"""

import pandas as pd


class DatasetLoader:
    """Class DatasetLoader to load the data with file_path"""

    def __init__(self, base_path):
        """
        Constructor to initialize the DatasetLoader instance with a base path.

        Parameters:
        - base_path (str): The base path where the datasets are located.
        """
        self.base_path = base_path

    def load_data_invt(self):
        """Load the inventory dataset."""
        file_path = f"{self.base_path}/Metro_invt_fs_uc_sfrcondo_sm_month (2).csv"
        return pd.read_csv(file_path)

    def load_data_doz_pending(self):
        """Load the dataset for Days on Zillow Pending."""
        file_path = (
            f"{self.base_path}/Metro_mean_doz_pending_uc_sfrcondo_sm_month (2).csv"
        )
        return pd.read_csv(file_path)

    def load_data_sales_count(self):
        """Load the dataset for sales count."""
        file_path = f"{self.base_path}/Metro_sales_count_now_uc_sfrcondo_month (2).csv"
        return pd.read_csv(file_path)

    def load_data_home_value_growth(self):
        """Load the dataset for home value growth."""
        file_path = f"{self.base_path}/Metro_zhvf_growth_uc_sfrcondo_tier_0.33_0.67_sm_sa_month (1).csv"
        return pd.read_csv(file_path)

    def load_data_home_value_index(self):
        """Load the dataset for home value index."""
        file_path = f"{self.base_path}/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month (1).csv"
        return pd.read_csv(file_path)

    def load_data_rental_index(self):
        """Load the dataset for rental index."""
        file_path = f"{self.base_path}/Metro_zori_uc_sfrcondomfr_sm_month (2).csv"
        return pd.read_csv(file_path)

    def load_csv(self, file_path):
        """Load a generic CSV file."""
        return pd.read_csv(file_path)

    def filter_and_reshape_us_data(self, dataframe, drop_columns):
        """
        Filter and reshape data for the United States.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.
        - drop_columns (list): Columns to drop from the input DataFrame.

        Returns:
        pd.DataFrame: Reshaped data for the United States.
        """
        filtered_data = dataframe[dataframe["RegionName"] == "United States"].drop(
            columns=drop_columns
        )
        reshaped_data = pd.melt(
            filtered_data, var_name="Date", value_name=dataframe.name
        ).sort_values(by="Date")
        reshaped_data["Date"] = pd.to_datetime(reshaped_data["Date"])
        return reshaped_data
