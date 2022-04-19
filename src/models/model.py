import math
import tensorflow as tf
import numpy as np
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score
from models.linear_regression import LinearRegressor
from models.random_forest import RandomForest
from models.neural_network import NeuralNetwork

from utils import mse_by_category, r2_by_category, get_model_weights, compare_prediction_to_actual


def mse(predicted, actual):
    print(np.array(predicted).shape, np.array(actual).shape)
    err = np.square(predicted - actual)
    print(err.shape)
    err = np.sum(err, axis=1)
    d = np.abs(err - np.median(err))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    err = err[s < 6.]
    print(err.shape)
    err = np.average(err, axis=0)
    print(err.shape)
    return err


class Model:
    def __init__(self, exclude: list = [], verbose: bool = False):
        self.exclude = exclude
        self.hyperparams: dict = {}
        self.models: dict = {}
        self.weights: dict = {}
        self.verbose: bool = verbose

    def fit(self, x_train, y_train):
        self.hyperparams = {
            "LiRe": {},
            "RaFo": {},
            "NeNe": {'optimizer': 'adam',
                     'loss': tf.keras.losses.MeanSquaredError(),
                     'metrics': ['accuracy'],
                     'inp_shape': np.array(x_train).shape[1:],
                     'out_shape': np.array(y_train).shape[1],
                     'epochs': 14}
        }
        self.models = {
            "LiRe": LinearRegressor('LinearRegressor', **(self.hyperparams["LiRe"])),
            "RaFo": RandomForest('RandomForest', **(self.hyperparams["RaFo"])),
            "NeNe": NeuralNetwork('NeuralNetwork', **(self.hyperparams["NeNe"]))
        }

        for key in self.exclude:
            del self.models[key]
            del self.hyperparams[key]

        for key, model in self.models.items():
            if self.verbose:
                logger.info(f'Training {model.name}')
            model.fit(x_train, y_train)
            if self.verbose:
                logger.info(f'Training completed for {model.name}')

        r2 = []
        for key in self.models.keys():
            predictions = self.models[key].predict(x_train)
            r2 += [r2_by_category(predictions, y_train)]

        weights = get_model_weights(np.array(r2))
        for ind, key in enumerate(self.models.keys()):
            self.weights[key] = weights[ind]

        return self

    def predict(self, x_test, y_test):
        total_predictions = []

        for key, model in self.models.items():
            prediction = np.array(model.predict(x_test))
            prediction = np.clip(prediction, -1, 1)
            
            if self.verbose:
                logger.info(f'Testing {model.name}')
                err = mse(prediction, y_test)
                score = model.score(x_test, y_test)
                logger.success(f'{model.name} has {err=} and {score=}')
                compare_prediction_to_actual(prediction[0], y_test[0], fname=model.name)

            prediction = [list(np.multiply(pred, self.weights[key])) for pred in prediction]
            total_predictions += [list(prediction)]

        total_predictions = np.sum(total_predictions, axis=0)
        if self.verbose:
            err = mse(total_predictions, y_test)
            r2 = r2_score(y_test, total_predictions)
            logger.info(f'Aggregate Model has {err=} and {r2=}')
        compare_prediction_to_actual(total_predictions[0], y_test[0], fname='aggregate')
