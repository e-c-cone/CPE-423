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
from processing.election_data import generate_dataset
from utils import compare_prediction_to_actual
from models.model import Model


def arguments():
    parser = argparse.ArgumentParser(description="BEEGUS")

    parser.add_argument("-v", "--verbose", default=False, action="store_true",
                        help="Include warning data")
    parser.add_argument("-rd", "--reload_data", default=False, action="store_true",
                        help="Reload data instead of using data saved on file")

    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    # utils.find_possible_categories()
    # exit()
    verbose = args.verbose

    # Filter data to be after specified cutoff year
    cutoff_year = 1990

    ###  Generate Election Data  ###
    # Here we clean the data, and reshape it using pandas dataframes to be an appropriate shape for input into the
    # Linear Regression algorithm. We start by generating the possible outputs (percentage of votes for each party)
    # and continue by adding the GDP and Income data. This is not expected to be very accurate.
    x, y, keys = generate_dataset(cutoff_year, args.verbose, args.reload_data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = Model(verbose=verbose).fit(x_train, y_train)
    model.predict(x_test, y_test)

    ###  Train Models ###
    # LiRe = LinearRegressor('LinearRegressor')
    # LiRe.fit(x_train, y_train)
    # LiRe.mse_by_category(x_train, y_train)

    # all_preds = []
    #
    # logger.info(f'Beginning Linear Regression Training')
    # LiR = LinearRegression().fit(x_train, y_train)
    # predictions = LiR.predict(x_test)
    # print(predictions.shape)
    # mse = mean_squared_error(predictions, y_test)
    # score = LiR.score(x_test, y_test)/len(x_test)
    # if args.verbose:
    #     logger.info(f'First predictions')
    #     print(f'\n{predictions[:1]}')
    #     logger.info(f'Actual Values')
    #     print(f'\n{y_test[:1]}')
    # utils.compare_prediction_to_actual(predictions[0], y_test[0], 'LR')
    # logger.info(f'Mean Squared Error for Test Set: {mse}')
    # logger.info(f'R^2 Score: {score}')
    # logger.success(f'Linear Regression Completed')
    # all_preds += [predictions]
    #
    # logger.info(f'Beginning Random Forest Regression Training')
    # RFR = RandomForestRegressor().fit(x_train, y_train)
    # predictions = RFR.predict(x_test)
    # mse = mean_squared_error(prediction# logger.info(f'Beginning Neural Network Training')
    #     # x_train = np.array(x_train)
    #     # x_test = np.array(x_test)
    #     # y_train = np.array(y_train)
    #     # y_test = np.array(y_test)
    #     # model = tf.keras.models.Sequential([
    #     #     tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    #     #     Dense(256, activation='relu'),
    #     #     tf.keras.layers.Dropout(0.15),
    #     #     Dense(256),
    #     #     tf.keras.layers.Dropout(0.15),
    #     #     Dense(256),
    #     #     Dense(y_train.shape[1])
    #     # ])
    #     # loss = tf.keras.losses.MeanSquaredError()
    #     # model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    #     # model.fit(x_train, y_train, epochs=10)
    #     # predictions = model.predict(x_test)
    #     # mse = mean_squared_error(predictions, y_test)
    #     # score = model.evaluate(x_test, y_test)[1]
    #     # if args.verbose:
    #     #     logger.info(f'First predictions')
    #     #     [print(pred) for pred in predictions[:1]]
    #     #     logger.info(f'Actual Values')
    #     #     print(f'\n{y_test[:1]}')
    #     # utils.compare_prediction_to_actual(predictions[0], y_test[0], 'NN')
    #     # logger.success(f'Mean Squared Error for Test Set: {mse}')
    #     # logger.success(f'R^2 Score: {score}')
    #     # logger.success(f'Neural Network Training Completed')
    #     # all_preds += [predictions]s, y_test)
    # score = RFR.score(x_test, y_test)
    # if args.verbose:
    #     logger.info(f'First predictions')
    #     [print(pred) for pred in predictions[:1]]
    #     logger.info(f'Actual Values')
    #     print(f'\n{y_test[:1]}')
    # utils.compare_prediction_to_actual(predictions[0], y_test[0], 'RFR')
    # logger.success(f'Mean Squared Error for Test Set: {mse}')
    # logger.success(f'R^2 Score: {score}')
    # logger.success(f'Random Forest Regression Completed')
    # all_preds += [predictions]
    #

    #
    # predictions = np.average(np.average(np.array(all_preds), axis=0), axis=0)
    # print(predictions)
    # y_test = np.average(y_test, axis=0)
    # print(predictions.shape, y_test.shape)
    # utils.compare_prediction_to_actual(predy=predictions, actualy=y_test, fname='averaged')
    # # print(predictions)
    # # print(y_test)
    # mse = np.power(predictions - y_test, 2)
    # mse = zip(mse, [i for i in range(len(mse))])
    # mse = [i for _, i in sorted(mse)][:9]
    # print(mse)
    # sig_data = pd.read_csv(os.path.join('Votesmart', 'sig_agg', 'ALL_SIG_DATA.csv'))
    # print(np.array(sig_data['category_name_1'].unique())[mse])
    #
    # logger.info(f'Processing data for LSTM')
    # # print(x)
    # # print(y)
    # inp = Input(shape=(50, 10, len(x_train[0])))

    # covMatrix = np.cov(x, bias=True)
    # # sn.heatmap(covMatrix, annot=True, fmt='g')
    # # plt.show()
    # np.savetxt('cov.csv', covMatrix, delimiter=',')
