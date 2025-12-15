from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
class TenureBinner(BaseEstimator, TransformerMixin):
    def __init__(self, column='Tenure', bins=None, labels=None):
        self.column = column
        self.bins = bins if bins is not None else [0, 6, 15, 22, float('inf')]
        self.labels = labels if labels is not None else ['New', 'Developing', 'Established', 'Loyal']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = pd.DataFrame(X).copy()
        if self.column not in X_.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame.")

        if X_[self.column].isnull().any():
            X_[self.column] = X_[self.column].fillna(X_[self.column].median())

        binned = pd.cut(X_[self.column], bins=self.bins, labels=self.labels, include_lowest=True)
        return pd.DataFrame({f"{self.column}_bin": binned})

    def get_feature_names_out(self, input_features=None):
        return [f"{self.column}_bin"]
