import os
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from loguru import logger
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import utils
import processing.load_data as load_data
from processing.election_data import data_generator
from utils import compare_prediction_to_actual
from models.model import Model


def arguments():
    parser = argparse.ArgumentParser(description="BEEGUS")

    parser.add_argument("-v", "--verbose", default=False, action="store_true",
                        help="Include warning data")
    parser.add_argument("-rd", "--reload_data", default=False, action="store_true",
                        help="Reload data instead of using data saved on file")
    parser.add_argument("-ex", "--exclude", action='store', help="Excludes a model")
    parser.add_argument("-cy", "--cutoff_year", action='store', default=2010, type=int,
                        help="Sets end year for model training")
    parser.add_argument("-t", "--train", action='store_true', default=False,
                        help="Indicates whether to train a model from scratch or to use a pretrained version")

    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    verbose = args.verbose

    ###  Generate Election Data  ###
    # Here we clean the data, and reshape it using pandas dataframes to be an appropriate shape for input into the
    # Linear Regression algorithm. We start by generating the possible outputs (percentage of votes for each party)
    # and continue by adding the GDP and Income data. This is not expected to be very accurate.

    # utils.generate_ids_from_cand_dir()
    dg = data_generator(args.cutoff_year, verbose, args.reload_data)
    x, y, additional_data, keys = dg.generate_dataset()

    train_inds = [(int(key.split('_')[1]) < int(args.cutoff_year)) for key in keys]
    test_inds = [not ind for ind in train_inds]

    x = np.array(x)
    y = np.array(y)

    # additional_data.to_csv('TEST.csv')
    x_train = x[train_inds]
    y_train = y[train_inds]
    x_test = x[test_inds]
    y_test = y[test_inds]
    keys = np.array(keys)[test_inds]
    additional_data = pd.DataFrame(additional_data).set_index('election_id').iloc[test_inds]

    if args.exclude:
        model = Model(inp_shape=np.array(x_train).shape[1:], out_shape=np.array(y_train).shape[1],
                      exclude=[args.exclude], verbose=verbose)
    else:
        model = Model(inp_shape=np.array(x_train).shape[1:], out_shape=np.array(y_train).shape[1], verbose=verbose)

    if args.train:
        model.fit(x_train, y_train)
        model.save()
    else:
        model.load()

    model.predict(x_test, y_test, additional_data, keys)
