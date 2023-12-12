# regression_analysis.py
"""

This module defines a class, RegressionModeler, for preparing data, training regression models,
evaluating their performance, creating lag features, dropping missing values, and performing feature selection.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


class RegressionModeler:
    """
    Class for preparing data, training regression models, and performing feature engineering.
    """

    def prepare_data(self, df1, df2, on_columns, feature_columns, target_column):
        """
        Prepares data by merging two DataFrames, selecting relevant columns, and handling missing values.

        Parameters
        ----------
        df1 : DataFrame
            The first DataFrame to be merged.
        df2 : DataFrame
            The second DataFrame to be merged.
        on_columns : list
            Columns used for merging the DataFrames.
        feature_columns : list
            Columns used as features in the regression model.
        target_column : str
            The target variable column.

        Returns
        -------
        tuple
            Features (X) and target variable (y).
        """
        merged_data = pd.merge(df1, df2, on=on_columns, how="inner")
        regression_data = merged_data[feature_columns + [target_column]].dropna()
        X = regression_data[feature_columns]
        y = regression_data[target_column]
        return X, y

    def split_data(self, X, y, test_size, random_state):
        """
        Splits data into training and testing sets.

        Parameters
        ----------
        X : DataFrame
            Features.
        y : Series
            Target variable.
        test_size : float
            Proportion of the dataset to include in the test split.
        random_state : int
            Seed for random state.

        Returns
        -------
        tuple
            Training and testing sets for features and target variable.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_random_forest(self, X_train, y_train, n_estimators, random_state):
        """
        Trains a Random Forest regression model.

        Parameters
        ----------
        X_train : DataFrame
            Training data features.
        y_train : Series
            Training data labels.
        n_estimators : int
            Number of trees in the forest.
        random_state : int
            Seed for random state.

        Returns
        -------
        RandomForestRegressor
            Trained Random Forest model.
        """
        rf_regressor = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
        rf_regressor.fit(X_train, y_train)
        return rf_regressor

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluates the performance of a regression model using Mean Squared Error and R-squared.

        Parameters
        ----------
        model : object
            Trained machine learning model.
        X_test : DataFrame
            Test data features.
        y_test : Series
            Test data labels.

        Returns
        -------
        tuple
            Mean Squared Error and R-squared.
        """
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def create_lag_features(self, dataframe, column_name, lag_number):
        """
        Creates lag features for a given column in a DataFrame.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.
        column_name : str
            Name of the column for which lag features will be created.
        lag_number : int
            Number of time periods to lag.

        Returns
        -------
        DataFrame
            DataFrame with lag features.
        """
        lag_column_name = f"{column_name}_Lag{lag_number}"
        dataframe[lag_column_name] = dataframe[column_name].shift(lag_number)
        return dataframe

    def drop_na_values(self, dataframe):
        """
        Drops rows with missing values from a DataFrame.

        Parameters
        ----------
        dataframe : DataFrame
            Input DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with missing values dropped.
        """
        dataframe.dropna(inplace=True)
        return dataframe

    def feature_selection_rfe(self, X, y, n_features):
        """
        Performs feature selection using Recursive Feature Elimination (RFE).

        Parameters
        ----------
        X : DataFrame
            Features.
        y : Series
            Target variable.
        n_features : int
            Number of features to select.

        Returns
        -------
        list
            List of tuples containing feature names, rankings, and support status.
        """
        linear_reg = LinearRegression()
        rfe = RFE(estimator=linear_reg, n_features_to_select=n_features)
        rfe = rfe.fit(X, y)
        feature_ranking = list(zip(X.columns, rfe.ranking_, rfe.support_))
        return feature_ranking
