import os
import math
import random
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score
from models.linear_regression import LinearRegressor
from models.random_forest import RandomForest
from models.neural_network import NeuralNetwork

from utils import mse_by_category, r2_by_category, get_model_weights, compare_prediction_to_actual


def mse(predicted, actual):
    err = np.square(predicted - actual)
    err = np.sum(err, axis=1)
    d = np.abs(err - np.median(err))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    err = err[s < 6.]
    err = np.average(err, axis=0)
    return err


class Model:
    def __init__(self, inp_shape, out_shape, exclude: list = [], verbose: bool = False):
        self.exclude = exclude
        self.weights: dict = {}
        self.verbose: bool = verbose
        self.save_dir: str = os.path.join('src', 'models', 'saved_models')
        self.hyperparams = {
            "LiRe": {'fname': 'linear_regressor',
                     'dir': self.save_dir},
            "RaFo": {'fname': 'random_forest',
                     'dir': self.save_dir},
            "NeNe": {'fname': 'neural_network',
                     'dir': self.save_dir,
                     'optimizer': 'adam',
                     'loss': tf.keras.losses.MeanSquaredError(),
                     'inp_shape': inp_shape,
                     'out_shape': out_shape,
                     'metrics': ['accuracy'],
                     'epochs': 15}
        }
        self.models = {
            "LiRe": LinearRegressor('LinearRegressor', **(self.hyperparams["LiRe"])),
            "RaFo": RandomForest('RandomForest', **(self.hyperparams["RaFo"])),
            "NeNe": NeuralNetwork('NeuralNetwork', **(self.hyperparams["NeNe"]))
        }

        for key in self.exclude:
            del self.models[key]
            del self.hyperparams[key]

    def fit(self, x_train, y_train, plot=False):
        for key, model in self.models.items():
            if self.verbose:
                logger.info(f'Training {model.name}')
            model.fit(x_train, y_train)
            if self.verbose:
                logger.info(f'Training completed for {model.name}')

        r2 = []
        for model in self.models.values():
            predictions = model.predict(x_train)
            r2 += [r2_by_category(predictions, y_train)]

        weights = get_model_weights(np.array(r2))
        for ind, key in enumerate(self.models.keys()):
            self.weights[key] = weights[ind].tolist()
        agg_prediction = self.predict(x_train, y_train, verbose=False)
        r2 += [r2_by_category(agg_prediction, y_train)]

        if plot:
            self.stats_by_category(x_train, y_train, 'Training')

        return self

    def predict(self, x_test, y_test, verbose=True):
        if verbose:
            verbose = self.verbose
        total_predictions = []
        ind = random.randint(0, len(x_test) - 1)

        for key, model in self.models.items():
            prediction = np.array(model.predict(x_test))
            prediction = np.clip(prediction, -1, 1)

            if verbose:
                logger.info(f'Testing {model.name}')
                err = mse(prediction, y_test)
                score = model.score(x_test, y_test)
                logger.success(f'{model.name} has {err=} and {score=}')

            compare_prediction_to_actual(prediction[ind], y_test[ind], fname=model.name)
            prediction = [list(np.multiply(pred, self.weights[key])) for pred in prediction]
            total_predictions += [list(prediction)]

        total_predictions = np.sum(total_predictions, axis=0)
        if verbose:
            err = mse(total_predictions, y_test)
            r2 = r2_score(y_test, total_predictions)
            logger.info(f'Aggregate Model has {err=} and {r2=}')
        compare_prediction_to_actual(total_predictions[ind], y_test[ind], fname='aggregate')

        return total_predictions

    def stats_by_category(self, x, y, type_of_test: str = 'TypeUnknown'):
        r2 = []
        mserr = []
        for model in self.models.values():
            predictions = model.predict(x)
            r2 += [r2_by_category(predictions, y)]
            mserr += [mse_by_category(predictions, y)]

        agg_prediction = self.predict(x, y, verbose=False)
        r2 += [r2_by_category(agg_prediction, y)]
        mserr += [mse_by_category(agg_prediction, y)]

        names = []
        for i, model_key in enumerate(self.models.keys()):
            plt.plot([j for j in range(len(r2[i]))], r2[i])
            names += [self.models[model_key].name]
        plt.plot([j for j in range(len(r2[-1]))], r2[-1])
        names += ['Aggregate']
        plt.legend(names)
        plt.title(f'{type_of_test} R^2 by Rating Category for each Model')
        plt.ylabel('R^2')
        plt.xlabel('Rating Category')
        plt.show()
        plt.clf()

        names = []
        for i, model_key in enumerate(self.models.keys()):
            plt.plot([j for j in range(len(mserr[i]))], mserr[i])
            names += [self.models[model_key].name]
        plt.plot([j for j in range(len(mserr[-1]))], mserr[-1])
        names += ['Aggregate']
        plt.legend(names)
        plt.title(f'{type_of_test} MSE by Rating Category for each Model')
        plt.ylabel('MSE')
        plt.xlabel('Rating Category')
        plt.show()
        plt.clf()

    def predict_last_year(self, x_test, y_test, additional_data: pd.DataFrame, keys: np.array):   # TODO
        total_predictions = self.predict(x_test, y_test)

        if self.verbose:
            self.stats_by_category(x_test, y_test, 'Testing')

        additional_data = additional_data[[type(cell) != str for cell in list(additional_data['loser_perc'])]]
        prediction_results_for_last_year = pd.DataFrame.from_dict({'election_id': keys.tolist(),
                                                                   'x_test': x_test.tolist(),
                                                                   'y_test': y_test.tolist(),
                                                                   'prediction': total_predictions.tolist()})
        prediction_results_for_last_year = prediction_results_for_last_year.set_index('election_id')
        prediction_results_for_last_year = pd.merge(prediction_results_for_last_year, additional_data,
                                                    left_index=True, right_index=True)
        # print(prediction_results_for_last_year)
        prediction_results_for_last_year.to_csv('pred_results_for_last_year.csv')

        prediction_results_for_last_year = prediction_results_for_last_year[
            prediction_results_for_last_year['winner_perc'].apply(type) != str]
        prediction_results_for_last_year = prediction_results_for_last_year[
            prediction_results_for_last_year['loser_perc'].apply(type) != str]
        predicted_ratings = np.array(prediction_results_for_last_year['prediction'])
        winner_ratings = prediction_results_for_last_year['y_test']
        loser_ratings = prediction_results_for_last_year['second_best_ratings']
        winner_ratings = np.array([np.array(rating).reshape((72)) for rating in winner_ratings])
        loser_ratings = np.array([np.array(rating).reshape((72)) for rating in loser_ratings])
        predicted_ratings = np.array([np.array(rating).reshape((72)) for rating in predicted_ratings])
        # print(f'{predicted_ratings.shape=}, {loser_ratings.shape=}, {winner_ratings.shape=}')

        winner_dif = np.subtract(predicted_ratings, winner_ratings)
        winner_dif = np.sum(np.square(winner_dif), axis=1)
        loser_dif = np.subtract(predicted_ratings, loser_ratings)
        loser_dif = np.sum(np.square(loser_dif), axis=1)
        prediction_results_for_last_year['winner_dif'] = winner_dif.tolist()
        prediction_results_for_last_year['loser_dif'] = loser_dif.tolist()

        tot_p = prediction_results_for_last_year['winner_perc'] + prediction_results_for_last_year['loser_perc']
        prediction_results_for_last_year['winner_perc'] = prediction_results_for_last_year['winner_perc'] / tot_p
        prediction_results_for_last_year['loser_perc'] = prediction_results_for_last_year['loser_perc'] / tot_p

        # Returns and saves the predictions for the last year. This DataFrame has the columns:
        # columns = ['election_id', 'x_test', 'y_test', 'second_best_ratings', 'winner_perc', 'loser_perc',
        #            'winner_dif', 'loser_perc']
        # Where election_id is the index, x_test is the test input as a list and y_test is the ratings as a list
        prediction_results_for_last_year.to_csv('pred_results_for_last_year.csv')
        logger.success(f'Results predicted for most recent available year.')
        return prediction_results_for_last_year

    def save(self):
        logger.info(f'Saving models to file...')
        for model in self.models.values():
            model.save()
        hyperparameters = {'exclude': self.exclude,
                           'weights': self.weights,
                           'verbose': self.verbose}
        with open(os.path.join(self.save_dir, 'aggregate_model_hyperparameters.json'), 'w', encoding='utf-8') as f:
            json.dump(hyperparameters, f, indent=4)
        logger.success(f'Models saved successfully')

    def load(self):
        logger.info(f'Loading models from file...')
        for model in self.models.values():
            model.load()

        with open(os.path.join(self.save_dir, 'aggregate_model_hyperparameters.json')) as f:
            hyperparameters = dict(json.load(f))
        for key, val in hyperparameters.items():
            self.__setattr__(key, val)
        logger.success(f'Models loaded successfully')
