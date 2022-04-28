import os
import pickle
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class LinearRegressor:
    def __init__(self, name: str, **kwargs):
        self.__dict__.update(kwargs)
        self.model = LinearRegression()
        self.name = name

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def score(self, x_test, y_test):
        return self.model.score(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)

    def save(self):
        with open(os.path.join(self.dir, self.fname + '.pkl'), 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        self.model = pickle.load(open(os.path.join(self.dir, self.fname+'.pkl'), 'rb'))
