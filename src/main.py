import os
import argparse
# import tensorflow as tf
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def arguments():
    parser = argparse.ArgumentParser(description="BEEGUS")

    parser.add_argument("-p19", "--post1997", default=True, action="store_true",
                        help="Determine whether to truncate data to post 1997")

    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()

    # Filter data to be after specified cutoff date according to argument
    cutoff_year = 1900
    if args.post1997:
        cutoff_year = 1997

    ###  Load data from data directory  ###
    # Load representative data; this data will serve as the basis for the rest of the data
    # It will be processed by converting each election into a unique identifier in the form of 'year_stateName'
    # This can then be used to pull data from other sources where the year and state name match
    house_rep = pd.read_csv(os.path.join('data', '1976_2020_house.csv'), sep=',', encoding='latin-1').dropna()
    house_rep_cols = ["year", "state", "state_po", "state_fips", "state_cen", "state_ic", "office", "district", "stage",
                      "special", "candidate", "party", "candidatevotes", "totalvotes"]
    for col in house_rep.columns:
        if col not in house_rep_cols:
            house_rep.drop(col, axis=1, inplace=True)
    house_rep = house_rep[house_rep['year'] >= cutoff_year]
    string_dtypes = house_rep.convert_dtypes().select_dtypes("string")
    house_rep[string_dtypes.columns] = string_dtypes.apply(lambda x: x.str.lower())
    logger.success(f"House Representatives loaded")

    # GDP by state by year
    GDP_by_state = pd.read_csv(os.path.join('data', 'SAGDP1__ALL_AREAS_1997_2020.csv'), sep=',', encoding='latin-1')
    for i, col in enumerate(GDP_by_state.columns):
        if i > 7 and int(col) < cutoff_year:
            GDP_by_state.drop(col, axis=1, inplace=True)
    string_dtypes = GDP_by_state.convert_dtypes().select_dtypes("string")
    GDP_by_state[string_dtypes.columns] = string_dtypes.apply(lambda x: x.str.lower())
    logger.success("GDP by State Loaded")

    # Personal Income by state by year
    personal_income_by_state = pd.read_csv(os.path.join('data', 'SAINC1__ALL_AREAS_1929_2020.csv'), sep=',', encoding='latin-1')
    for i, col in enumerate(personal_income_by_state.columns):
        if i > 7 and int(col) < cutoff_year:
            personal_income_by_state.drop(col, axis=1, inplace=True)
    string_dtypes = personal_income_by_state.convert_dtypes().select_dtypes("string")
    personal_income_by_state[string_dtypes.columns] = string_dtypes.apply(lambda x: x.str.lower())
    logger.success("Personal Income by State Loaded")

    ###  Generate Election Data  ###
    # Here we clean the data, and reshape it using pandas dataframes to be an appropriate shape for input into the
    # Linear Regression algorithm. We start by generating the possible outputs (percentage of votes for each party)
    # and continue by adding the GDP and Income data. This is not expected to be very accurate.
    house_rep['year_and_state'] = house_rep['year'].astype(str)+'_'+house_rep['state']
    elections = pd.DataFrame(house_rep['year_and_state'].unique(), columns=['year_and_state']).set_index('year_and_state')
    X = ['percent_national_GDP', 'percent_national_income', 'percent_national_population']
    Y = list(house_rep['party'].unique())
    elections[X] = 0.
    elections[Y] = 0.

    for index, row in house_rep.iterrows():
        elections.at[row['year_and_state'], row['party']] = row['candidatevotes']/row['totalvotes']
    for index, row in elections.iterrows():
        try:
            state_annual_GDP = float(GDP_by_state.loc[(GDP_by_state['GeoName'] == index[5:]) & (GDP_by_state['LineCode'] == 3), index[:4]])
            US_annual_GDP = float(GDP_by_state.loc[(GDP_by_state['GeoName'] == 'united states') & (GDP_by_state['LineCode'] == 3), index[:4]])
            percent_national_GDP = state_annual_GDP/US_annual_GDP
            elections.at[index, 'percent_national_GDP'] = percent_national_GDP

            state_annual_income = float(personal_income_by_state.loc[(personal_income_by_state['GeoName'] == index[5:])
                                                                     & (personal_income_by_state['LineCode'] == 1), index[:4]])
            US_annual_income = float(personal_income_by_state.loc[(personal_income_by_state['GeoName'] == 'united states')
                                                                  & (personal_income_by_state['LineCode'] == 1), index[:4]])
            percent_national_income = state_annual_income / US_annual_income
            elections.at[index, 'percent_national_income'] = percent_national_income

            state_population = float(personal_income_by_state.loc[(personal_income_by_state['GeoName'] == index[5:])
                                                                     & (personal_income_by_state['LineCode'] == 2), index[:4]])
            US_population = float(personal_income_by_state.loc[(personal_income_by_state['GeoName'] == 'united states')
                                                               & (personal_income_by_state['LineCode'] == 2), index[:4]])
            percent_national_population = state_population / US_population
            elections.at[index, 'percent_national_population'] = percent_national_population

        except:
            print(index[5:], index[:4])
            print(personal_income_by_state.loc[(personal_income_by_state['GeoName'] == index[5:]) & (personal_income_by_state['LineCode'] == 1), index[:4]])
            print(personal_income_by_state.loc[(personal_income_by_state['GeoName'] == 'united states') & (personal_income_by_state['LineCode'] == 1), index[:4]])
            exit()
    x_train, x_test, y_train, y_test = train_test_split(elections[X], elections[Y], test_size=0.3)

    logger.info(f'Beginning Linear Regression Training')
    LR = LinearRegression().fit(x_train, y_train)
    predictions = LR.predict(x_test)
    mse = mean_squared_error(predictions, y_test)
    score = LR.score(x_test, y_test)
    logger.info(f'First predictions')
    [print(pred[:4]) for pred in predictions[:4]]
    logger.info(f'Actual Values')
    print(f'\n{y_test[:4]}')
    logger.info(f'Mean Squared Error for Test Set: {mse}')
    logger.info(f'R^2 Score: {score}')
    logger.success(f'Linear Regression Completed')

    logger.info(f'Beginning Random Forest Regression Training')
    RFR = RandomForestRegressor().fit(x_train, y_train)
    predictions = RFR.predict(x_test)
    mse = mean_squared_error(predictions, y_test)
    score = RFR.score(x_test, y_test)
    logger.info(f'First predictions')
    [print(pred[:4]) for pred in predictions[:4]]
    logger.info(f'Actual Values')
    print(f'\n{y_test[:4]}')
    logger.success(f'Mean Squared Error for Test Set: {mse}')
    logger.success(f'R^2 Score: {score}')
    logger.success(f'Random Forest Regression Completed')

    house_rep['candidate'].to_csv('candidates.csv', columns=['candidate'], index=False)


