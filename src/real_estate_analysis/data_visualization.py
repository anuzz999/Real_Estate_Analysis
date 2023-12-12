# data_visualization.py

"""
This module provides a class, DataVisualizer, for visualizing data using Seaborn and Matplotlib.
It includes methods for plotting line graphs and time series data.

"""
import seaborn as sns
import matplotlib.pyplot as plt


class DataVisualizer:
    """
    A class for visualizing data using Seaborn and Matplotlib.

    Methods
    -------
    plot_line(data, x, y, hue, title, x_label, y_label, figsize=(15, 8), marker="o")
        Plots a line graph using Seaborn.

    plot_time_series(data, title, x_label, y_label, figsize=(15, 6))
        Plots time series data using Seaborn.
    """

    def plot_line(
        data, x, y, hue, title, x_label, y_label, figsize=(15, 8), marker="o"
    ):
        """
        Plots a line graph using Seaborn.

        Parameters
        ----------
        data : DataFrame
            The input data to be visualized.
        x : str
            The column name for the x-axis.
        y : str
            The column name for the y-axis.
        hue : str
            The column name for color differentiation (optional).
        title : str
            The title of the plot.
        x_label : str
            The label for the x-axis.
        y_label : str
            The label for the y-axis.
        figsize : tuple, optional
            The size of the figure (width, height), by default (15, 8).
        marker : str, optional
            The marker style for data points on the line, by default "o".

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        sns.lineplot(data=data, x=x, y=y, hue=hue, marker=marker)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.show()

    def plot_time_series(self, data, title, x_label, y_label, figsize=(15, 6)):
        """
        Plots time series data using Seaborn.

        Parameters
        ----------
        data : DataFrame
            The input time series data to be visualized.
        title : str
            The title of the plot.
        x_label : str
            The label for the x-axis.
        y_label : str
            The label for the y-axis.
        figsize : tuple, optional
            The size of the figure (width, height), by default (15, 6).

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        for column in data.columns[1:]:
            sns.lineplot(x=data["Date"], y=data[column], label=column)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    def plot_heatmap(self, data, title, figsize=(6, 4)):
        """
        Plots a heatmap to visualize the correlation matrix of the input data.

        Parameters
        ----------
        data : DataFrame
            The input data for which the correlation heatmap will be generated.
        title : str
            The title of the heatmap.
        figsize : tuple, optional
            The size of the figure (width, height), by default (6, 4).

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
        plt.title(title)
        plt.show()

    def plot_bar(self, data, title, x_label, y_label, color, alpha=0.6):
        """
        Plots a bar chart for the input data.

        Parameters
        ----------
        data : DataFrame
            The input data for which the bar chart will be generated.
        title : str
            The title of the bar chart.
        x_label : str
            The label for the x-axis.
        y_label : str
            The label for the y-axis.
        color : str
            The color of the bars.
        alpha : float, optional
            The transparency of the bars, by default 0.6.

        Returns
        -------
        None
        """
        plt.figure(figsize=(15, 6))
        data.plot(kind="bar", color=color, alpha=alpha)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.show()

    def plot_line_seasonal(self, data, title, x_label, y_label):
        """
        Plots a seasonal line plot for the input data.

        Parameters
        ----------
        data : DataFrame
            The input data containing temporal information for the x-axis.
        title : str
            The title of the seasonal line plot.
        x_label : str
            The label for the x-axis.
        y_label : str
            The label for the y-axis.

        Returns
        -------
        None
        """
        plt.figure(figsize=(15, 8))
        for column in data.columns[1:]:  # Assuming the first column is 'Month-Year'
            sns.lineplot(x=data["Month-Year"], y=data[column], label=column)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=45)  # Rotating x-axis labels for better readability
        plt.legend()
        plt.grid(True)
        plt.tight_layout()  # Adjusting layout
        plt.show()
