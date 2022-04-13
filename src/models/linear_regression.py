from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class LinearRegressor:
    def __init__(self, name: str, **kwargs):
        self.__dict__.update(kwargs)
        self.model = LinearRegression()
        self.name = name
        # self.scaler = StandardScaler()

    def fit(self, x_train, y_train, x=None):
        # if x:
        #     x_train = self.scaler.fit_transform(x_train)
        self.model.fit(x_train, y_train)

    def score(self, x_test, y_test):
        # x_test = self.scaler.transform(x_test)
        return self.model.score(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)
