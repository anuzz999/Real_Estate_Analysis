# machine_learning.py


"""

This module defines classes for building machine learning models and evaluating their performance.

"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class ModelBuilder:
    """
    Class for building machine learning models.

    Methods
    -------
    build_random_forest(X_train, y_train, n_estimators=100, random_state=0)
        Builds a Random Forest regression model.

    make_predictions(model, X_test)
        Makes predictions using the given model.

    extract_feature_importances(model, feature_names)
        Extracts feature importances from the given model.

    build_gradient_boosting(X_train, y_train, n_estimators=100, random_state=42)
        Builds a Gradient Boosting regression model.

    build_linear_regression(X_train, y_train)
        Builds a Linear Regression model.
    """

    def build_random_forest(X_train, y_train, n_estimators=100, random_state=0):
        """
        Builds a Random Forest regression model.

        Parameters
        ----------
        X_train : DataFrame
            Training data features.
        y_train : Series
            Training data labels.
        n_estimators : int, optional
            Number of trees in the forest, by default 100.
        random_state : int, optional
            Seed for random state, by default 0.

        Returns
        -------
        RandomForestRegressor
            Trained Random Forest model.
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
        model.fit(X_train, y_train)
        return model

    def make_predictions(model, X_test):
        """
        Makes predictions using the given model.

        Parameters
        ----------
        model : object
            Trained machine learning model.
        X_test : DataFrame
            Test data features.

        Returns
        -------
        array
            Model predictions.
        """
        return model.predict(X_test)

    def extract_feature_importances(model, feature_names):
        """
        Extracts feature importances from the given model.

        Parameters
        ----------
        model : object
            Trained machine learning model.
        feature_names : list
            List of feature names.

        Returns
        -------
        DataFrame
            DataFrame with feature names and their importances.
        """
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importances}
        )
        return importance_df.sort_values(by="Importance", ascending=False)

    def build_gradient_boosting(X_train, y_train, n_estimators=100, random_state=42):
        """
        Builds a Gradient Boosting regression model.

        Parameters
        ----------
        X_train : DataFrame
            Training data features.
        y_train : Series
            Training data labels.
        n_estimators : int, optional
            Number of boosting stages to be run, by default 100.
        random_state : int, optional
            Seed for random state, by default 42.

        Returns
        -------
        GradientBoostingRegressor
            Trained Gradient Boosting model.
        """
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
        model.fit(X_train, y_train)
        return model

    def build_linear_regression(X_train, y_train):
        """
        Builds a Linear Regression model.

        Parameters
        ----------
        X_train : DataFrame
            Training data features.
        y_train : Series
            Training data labels.

        Returns
        -------
        LinearRegression
            Trained Linear Regression model.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model


class ModelEvaluator:
    """
    Class for evaluating machine learning models.

    Methods
    -------
    evaluate_model(y_true, y_pred)
        Evaluates the performance of a model using Mean Absolute Error, Root Mean Squared Error, and R-squared.

    get_feature_importances(model, feature_names)
        Gets feature importances from the given model.

    extract_coefficients(model, feature_names)
        Extracts coefficients from the given linear regression model.
    """

    def evaluate_model(y_true, y_pred):
        """
        Evaluates the performance of a model using Mean Absolute Error, Root Mean Squared Error, and R-squared.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.

        Returns
        -------
        tuple
            Mean Absolute Error, Root Mean Squared Error, R-squared.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    def get_feature_importances(model, feature_names):
        """
        Gets feature importances from the given model.

        Parameters
        ----------
        model : object
            Trained machine learning model.
        feature_names : list
            List of feature names.

        Returns
        -------
        DataFrame
            DataFrame with feature names and their importances.
        """
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importances}
        )
        return importance_df.sort_values(by="Importance", ascending=False)

    def extract_coefficients(model, feature_names):
        """
        Extracts coefficients from the given linear regression model.

        Parameters
        ----------
        model : object
            Trained Linear Regression model.
        feature_names : list
            List of feature names.

        Returns
        -------
        DataFrame
            DataFrame with feature names and their coefficients.
        """
        coefficients = pd.DataFrame(
            {"Feature": feature_names, "Coefficient": model.coef_}
        )
        return coefficients.sort_values(by="Coefficient", ascending=False)
