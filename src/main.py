import os
import argparse
# import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import utils
import processing.load_data as load_data
from processing.election_data import generate_dataset


def arguments():
    parser = argparse.ArgumentParser(description="BEEGUS")

    parser.add_argument("-v", "--verbose", default=False, action="store_true",
                        help="Include warning data")
    parser.add_argument("-rd", "--reload_data", default=False, action="store_true",
                        help="Reload data instead of using data saved on file")

    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    # utils.cluster_interest_groups()
    # exit()

    # Filter data to be after specified cutoff year
    cutoff_year = 1990
    # house_reps = load_data.get_candidates(cutoff_year)
    # print(utils.get_proper_names(house_reps))
    # utils.generate_ids_from_cand_dir()

    # exit()

    # # GDP by state by year
    # GDP_by_state = pd.read_csv(os.path.join('data', 'SAGDP1__ALL_AREAS_1997_2020.csv'), sep=',', encoding='latin-1')
    # for i, col in enumerate(GDP_by_state.columns):
    #     if i > 7 and int(col) < cutoff_year:
    #         GDP_by_state.drop(col, axis=1, inplace=True)
    # string_dtypes = GDP_by_state.convert_dtypes().select_dtypes("string")
    # GDP_by_state[string_dtypes.columns] = string_dtypes.apply(lambda x: x.str.lower())
    # logger.success("GDP by State Loaded")

    # personal_income_by_state = load_data.get_personal_income(year=cutoff_year)
    # print(personal_income_by_state)

    ###  Generate Election Data  ###
    # Here we clean the data, and reshape it using pandas dataframes to be an appropriate shape for input into the
    # Linear Regression algorithm. We start by generating the possible outputs (percentage of votes for each party)
    # and continue by adding the GDP and Income data. This is not expected to be very accurate.

    x, y = generate_dataset(cutoff_year, args.verbose, args.reload_data)

    # house_rep['year_and_state'] = house_rep['year'].astype(str) + '_' + str(house_rep['state_fips'])
    # elections = pd.DataFrame(house_rep['year_and_state'].unique(), columns=['year_and_state']).set_index(
    #     'year_and_state')
    # X = ['percent_national_GDP', 'percent_national_income', 'percent_national_population']
    # Y = list(house_rep['party'].unique())
    # elections[X] = 0.
    # elections[Y] = 0.

    # for index, row in house_rep.iterrows():
    #     elections.at[row['year_and_state'], row['party']] = row['candidatevotes'] / row['totalvotes']
    # for index, row in elections.iterrows():
    #     try:
    #         # state_annual_GDP = float(
    #         #     GDP_by_state.loc[(GDP_by_state['GeoName'] == index[5:]) & (GDP_by_state['LineCode'] == 3), index[:4]])
    #         # US_annual_GDP = float(GDP_by_state.loc[(GDP_by_state['GeoName'] == 'united states') & (
    #         #         GDP_by_state['LineCode'] == 3), index[:4]])
    #         # percent_national_GDP = state_annual_GDP / US_annual_GDP
    #         # elections.at[index, 'percent_national_GDP'] = percent_national_GDP
    #
    #         state_annual_income = float(personal_income_by_state.loc[(personal_income_by_state['GeoName'] == index[5:])
    #                                                                  & (personal_income_by_state[
    #                                                                         'LineCode'] == 1), index[:4]])
    #         US_annual_income = float(
    #             personal_income_by_state.loc[(personal_income_by_state['GeoName'] == 'united states')
    #                                          & (personal_income_by_state['LineCode'] == 1), index[:4]])
    #         percent_national_income = state_annual_income / US_annual_income
    #         elections.at[index, 'percent_national_income'] = percent_national_income
    #
    #         state_population = float(personal_income_by_state.loc[(personal_income_by_state['GeoName'] == index[5:])
    #                                                               & (personal_income_by_state['LineCode'] == 2), index[
    #                                                                                                              :4]])
    #         US_population = float(personal_income_by_state.loc[(personal_income_by_state['GeoName'] == 'united states')
    #                                                            & (personal_income_by_state['LineCode'] == 2), index[
    #                                                                                                           :4]])
    #         percent_national_population = state_population / US_population
    #         elections.at[index, 'percent_national_population'] = percent_national_population
    #
    #     except:
    #         print(index[5:], index[:4])
    #         print(personal_income_by_state.loc[(personal_income_by_state['GeoName'] == index[5:]) & (
    #                 personal_income_by_state['LineCode'] == 1), index[:4]])
    #         print(personal_income_by_state.loc[(personal_income_by_state['GeoName'] == 'united states') & (
    #                 personal_income_by_state['LineCode'] == 1), index[:4]])
    #         exit()
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # logger.debug(len(x_train))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    logger.info(f'Beginning Linear Regression Training')
    LR = LinearRegression().fit(x_train, y_train)
    predictions = LR.predict(x_test)
    mse = mean_squared_error(predictions, y_test) / len(predictions)
    score = LR.score(x_test, y_test) / len(predictions)
    if args.verbose:
        logger.info(f'First predictions')
        print(f'\n{predictions[:1]}')
        logger.info(f'Actual Values')
        print(f'\n{y_test[:1]}')
    logger.info(f'Mean Squared Error for Test Set: {mse}')
    logger.info(f'R^2 Score: {score}')
    logger.success(f'Linear Regression Completed')

    logger.info(f'Beginning Random Forest Regression Training')
    RFR = RandomForestRegressor().fit(x_train, y_train)
    predictions = RFR.predict(x_test)
    mse = mean_squared_error(predictions, y_test)
    score = RFR.score(x_test, y_test)
    if args.verbose:
        logger.info(f'First predictions')
        [print(pred) for pred in predictions[:1]]
        logger.info(f'Actual Values')
        print(f'\n{y_test[:1]}')
    logger.success(f'Mean Squared Error for Test Set: {mse}')
    logger.success(f'R^2 Score: {score}')
    logger.success(f'Random Forest Regression Completed')

    covMatrix = np.cov(x, bias=True)
    # sn.heatmap(covMatrix, annot=True, fmt='g')
    # plt.show()
    np.savetxt('cov.csv', covMatrix, delimiter=',')
