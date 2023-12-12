# data_visualization.py

"""

This module defines a class, DataAnalyzer, for analyzing and visualizing data.

"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class DataAnalyzer:
    """
    Class for analyzing and visualizing data.

    Methods
    -------
    get_descriptive_stats(dataframe)
        Retrieves descriptive statistics for the DataFrame.

    aggregate_by_category(dataframe, category_column, aggregation_columns)
        Aggregates data by a specified category column.

    sample_addresses(dataframe, column_name, sample_size, seed=1)
        Samples addresses from a specified column.

    find_time_related_columns(dataframe)
        Finds columns related to time in the DataFrame.

    calculate_correlation_matrix(dataframe, numerical_columns)
        Calculates the correlation matrix for numerical columns.

    correlation_with_target(correlation_matrix, target_column)
        Calculates the correlation of numerical columns with the target column.

    aggregate_data(dataframe, group_by_column, agg_columns)
        Aggregates data by a specified column.
    """

    def get_descriptive_stats(dataframe):
        """
        Retrieves descriptive statistics for the DataFrame.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.

        Returns
        -------
        DataFrame
            Descriptive statistics for the DataFrame.
        """
        return dataframe.describe()

    def aggregate_by_category(dataframe, category_column, aggregation_columns):
        """
        Aggregates data by a specified category column.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        category_column : str
            Column used for grouping.
        aggregation_columns : list
            Columns for which the mean is calculated.

        Returns
        -------
        DataFrame
            Aggregated data by category.
        """
        aggregated_data = dataframe.groupby(category_column)[aggregation_columns].mean()
        return aggregated_data

    def sample_addresses(dataframe, column_name, sample_size, seed=1):
        """
        Samples addresses from a specified column.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        column_name : str
            Column containing addresses.
        sample_size : int
            Size of the address sample.
        seed : int, optional
            Seed for random state, by default 1.

        Returns
        -------
        Series
            Sampled addresses.
        """
        return dataframe[column_name].sample(sample_size, random_state=seed)

    def find_time_related_columns(dataframe):
        """
        Finds columns related to time in the DataFrame.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.

        Returns
        -------
        list
            List of columns related to time.
        """
        return [
            col
            for col in dataframe.columns
            if "year" in col.lower() or "date" in col.lower()
        ]

    def calculate_correlation_matrix(dataframe, numerical_columns):
        """
        Calculates the correlation matrix for numerical columns.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        numerical_columns : list
            Numerical columns for which the correlation matrix is calculated.

        Returns
        -------
        DataFrame
            Correlation matrix for numerical columns.
        """
        return dataframe[numerical_columns].corr()

    def correlation_with_target(correlation_matrix, target_column):
        """
        Calculates the correlation of numerical columns with the target column.

        Parameters
        ----------
        correlation_matrix : DataFrame
            Correlation matrix for numerical columns.
        target_column : str
            Target column for which correlation is calculated.

        Returns
        -------
        Series
            Correlation of numerical columns with the target column.
        """
        return correlation_matrix[target_column].sort_values(ascending=False)

    def aggregate_data(dataframe, group_by_column, agg_columns):
        """
        Aggregates data by a specified column.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        group_by_column : str
            Column used for grouping.
        agg_columns : list
            Columns for which the mean is calculated.

        Returns
        -------
        DataFrame
            Aggregated data by a specified column.
        """
        return dataframe.groupby(group_by_column)[agg_columns].mean().reset_index()

    def get_descriptive_stats_subset(dataframe, columns):
        """
        Retrieves descriptive statistics for a subset of columns.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        columns : list
            Columns for which descriptive statistics are calculated.

        Returns
        -------
        DataFrame
            Descriptive statistics for the subset of columns.
        """
        return dataframe[columns].describe()

    def add_price_category(
        dataframe, price_column, bins, labels, category_column="Price_Category"
    ):
        """
        Adds a price category column based on specified bins and labels.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        price_column : str
            Column used for defining price categories.
        bins : list
            Binning intervals for defining categories.
        labels : list
            Labels for the categories.
        category_column : str, optional
            Name of the new category column, by default "Price_Category".

        Returns
        -------
        DataFrame
            DataFrame with the added price category column.
        """
        if category_column not in dataframe.columns:
            dataframe[category_column] = pd.cut(
                dataframe[price_column], bins=bins, labels=labels
            )
        return dataframe

    def calculate_correlation_with_price(dataframe, target_column="List_price"):
        """
        Calculates the correlation of all columns with a specified target price column.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        target_column : str, optional
            Target price column for which correlation is calculated, by default "List_price".

        Returns
        -------
        Series
            Correlation of all columns with the target price column.
        """
        return dataframe.corr()[target_column].sort_values(ascending=False)

    def convert_categorical_to_numeric(dataframe, category_column):
        """
        Converts a categorical column to numeric by creating a new numeric column.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        category_column : str
            Categorical column to be converted.

        Returns
        -------
        DataFrame
            DataFrame with the added numeric representation of the categorical column.
        """
        dataframe[category_column + "_Numeric"] = dataframe[category_column].cat.codes
        return dataframe


class DataVisualizer:
    """
    Class for data visualization.

    Methods
    -------
    plot_distribution_grid(dataframe, column_details, figsize=(15, 10))
        Plots a grid of distributions for specified columns.

    plot_category_counts(dataframe, category_column, title="Category Counts", figsize=(10, 6))
        Plots counts of unique values in a categorical column.

    plot_feature_importances(importance_df, title="Feature Importances", figsize=(12, 8))
        Plots feature importances from a DataFrame.

    plot_heatmap(data, title, figsize=(15, 10), fmt=".2f")
        Plots a heatmap for a given data matrix.

    plot_histogram(data, title, xlabel, ylabel, bins=30, figsize=(10, 6))
        Plots a histogram for a given data column.

    plot_distribution_subplot(dataframe, column_details, nrows, ncols, figsize=(15, 10))
        Plots distribution subplots for specified columns.

    plot_category_distribution(dataframe, category_column, title="Distribution Across Categories", figsize=(8, 6))
        Plots the distribution of a categorical column.
    """

    def plot_distribution_grid(dataframe, column_details, figsize=(15, 10)):
        """
        Plots a grid of distributions for specified columns.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        column_details : list of tuples
            List of tuples containing column name, KDE flag, and bins (optional).
        figsize : tuple, optional
            Size of the figure, by default (15, 10).
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        for i, (col, kde, bins) in enumerate(column_details):
            # Check if bins is defined, if not, omit the bins parameter
            if bins is None:
                sns.histplot(dataframe[col], kde=kde, ax=axes[i // 2, i % 2])
            else:
                sns.histplot(dataframe[col], kde=kde, bins=bins, ax=axes[i // 2, i % 2])
            axes[i // 2, i % 2].set_title(f"Distribution of {col}")
            axes[i // 2, i % 2].set_xlabel(col)
            axes[i // 2, i % 2].set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_category_counts(
        dataframe, category_column, title="Category Counts", figsize=(10, 6)
    ):
        """
        Plots counts of unique values in a categorical column.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        category_column : str
            Categorical column for which counts are plotted.
        title : str, optional
            Title of the plot, by default "Category Counts".
        figsize : tuple, optional
            Size of the figure, by default (10, 6).
        """
        plt.figure(figsize=figsize)
        dataframe[category_column].value_counts().plot(kind="bar")
        plt.title(title)
        plt.xlabel(category_column)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    def plot_feature_importances(
        importance_df, title="Feature Importances", figsize=(12, 8)
    ):
        """
        Plots feature importances from a DataFrame.

        Parameters
        ----------
        importance_df : DataFrame
            DataFrame containing feature importances.
        title : str, optional
            Title of the plot, by default "Feature Importances".
        figsize : tuple, optional
            Size of the figure, by default (12, 8).
        """
        plt.figure(figsize=figsize)
        sns.barplot(x="Importance", y="Feature", data=importance_df)
        plt.title(title)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()

    def plot_heatmap(data, title, figsize=(15, 10), fmt=".2f"):
        """
        Plots a heatmap for a given data matrix.

        Parameters
        ----------
        data : DataFrame
            Input data matrix.
        title : str
            Title of the plot.
        figsize : tuple, optional
            Size of the figure, by default (15, 10).
        fmt : str, optional
            Format string for annotating cells, by default ".2f".
        """
        plt.figure(figsize=figsize)
        sns.heatmap(data, annot=True, cmap="coolwarm", fmt=fmt)
        plt.title(title)
        plt.show()

    def plot_histogram(data, title, xlabel, ylabel, bins=30, figsize=(10, 6)):
        """
        Plots a histogram for a given data column.

        Parameters
        ----------
        data : Series
            Input data column.
        title : str
            Title of the plot.
        xlabel : str
            Label for the x-axis.
        ylabel : str
            Label for the y-axis.
        bins : int, optional
            Number of bins for the histogram, by default 30.
        figsize : tuple, optional
            Size of the figure, by default (10, 6).
        """
        plt.figure(figsize=figsize)
        sns.histplot(data, bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_distribution_subplot(
        dataframe, column_details, nrows, ncols, figsize=(15, 10)
    ):
        """
        Plots distribution subplots for specified columns.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        column_details : list of tuples
            List of tuples containing column name and plot type ('hist' or 'count').
        nrows : int
            Number of rows in the subplot grid.
        ncols : int
            Number of columns in the subplot grid.
        figsize : tuple, optional
            Size of the figure, by default (15, 10).
        """
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, (col, plot_type) in enumerate(column_details):
            ax = axes[i // ncols, i % ncols]
            if plot_type == "hist":
                sns.histplot(dataframe[col], kde=True, ax=ax)
            elif plot_type == "count":
                sns.countplot(x=col, data=dataframe, ax=ax)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_category_distribution(
        dataframe,
        category_column,
        title="Distribution Across Categories",
        figsize=(8, 6),
    ):
        """
        Plots the distribution of a categorical column.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        category_column : str
            Categorical column for which distribution is plotted.
        title : str, optional
            Title of the plot, by default "Distribution Across Categories".
        figsize : tuple, optional
            Size of the figure, by default (8, 6).
        """
        plt.figure(figsize=figsize)
        sns.countplot(
            x=category_column,
            data=dataframe,
            order=dataframe[category_column].cat.categories,
        )
        plt.title(title)
        plt.xlabel(category_column)
        plt.ylabel("Number of Properties")
        plt.show()
