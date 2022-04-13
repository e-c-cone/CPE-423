from loguru import logger
from sklearn.ensemble import RandomForestRegressor


class RandomForest:
    def __init__(self, name: str, **kwargs):
        self.__dict__.update(kwargs)
        self.model = RandomForestRegressor()
        self.name = name

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def score(self, x_test, y_test):
        return self.model.score(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)