from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator

class CustomLinearModel(BaseEstimator):
    """
    Wrapper class for custom linear model
    """

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y=None):
        self.model.fit(X, y)

        return self

    def predict(self, X):
        predictions = self.model.predict(X)

        return predictions