import os
import re
import pandas as pd
import numpy as np
from loguru import logger

POSSIBLE_RATING_CATEGORIES = {'Social', 'Civil Liberties and Civil Rights', 'Socially Conservative', 'Religion',
                              'Socially Liberal', 'Drugs', 'Science, Technology and Communication', 'Conservative',
                              'Elections', 'Sexual Orientation and Gender Identity', 'Abortion', 'Agriculture and Food',
                              'Campaign Finance', 'Animals and Wildlife', 'Health Insurance', 'Transportation',
                              'Technology and Communication', 'Infrastructure', 'Foreign Aid', 'Marriage',
                              'Foreign Affairs', 'Energy', 'Natural Resources', 'Guns', 'Education',
                              'Business, Consumers, and Employees', 'Constitution', 'Military Personnel', 'Legal',
                              'Oil and Gas', 'Federal, State and Local Relations', 'Taxes', 'Business and Consumers',
                              'Gambling and Gaming', 'Arts, Entertainment, and History', 'Environment', 'Marijuana',
                              'Science', 'Minors and Children', 'Criminal Justice', 'Health and Health Care',
                              'Unemployed and Low-Income', 'Government Operations', 'Food Processing and Sales',
                              'Fiscally Liberal', 'Labor Unions', 'Senior Citizens', 'Impartial/Nonpartisan',
                              'Finance and Banking', 'Reproduction', 'Defense', 'Entitlements and the Safety Net',
                              'Fiscally Conservative', 'Family', 'Legislative Branch', 'Budget, Spending and Taxes',
                              'Employment and Affirmative Action', 'Women', 'Judicial Branch', 'Veterans', 'Trade',
                              'Immigration', 'Liberal', 'Housing and Property', 'Government Budget and Spending',
                              'Economy and Fiscal', 'Higher Education'}
TEMPLATE_RATING_DICT = {key: 0 for key in POSSIBLE_RATING_CATEGORIES}


def get_candidates(cutoff_year: int = 1900) -> pd.DataFrame:
    """
    Load data from data directory
    Load representative data; this data will serve as the basis for the rest of the data
    It will be processed by converting each election into a unique identifier in the form of 'year_stateName'
    This can then be used to pull data from other sources where the year and state name match
    :param cutoff_year:
    """
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
    # print(house_rep)
    return house_rep


def get_personal_income(cutoff_year: int = 1900) -> pd.DataFrame:
    """
    Load personal income data from directory and return as a dataframe
    :param cutoff_year:
    """
    personal_income_by_state = pd.read_csv(os.path.join('data', 'SAINC1__ALL_AREAS_1929_2020.csv'), sep=',',
                                           encoding='latin-1')
    for i, col in enumerate(personal_income_by_state.columns):
        if i > 7 and int(col) < cutoff_year:
            personal_income_by_state.drop(col, axis=1, inplace=True)
    string_dtypes = personal_income_by_state.convert_dtypes().select_dtypes("string")
    personal_income_by_state[string_dtypes.columns] = string_dtypes.apply(lambda x: x.str.lower())
    logger.success("Personal Income by State Loaded")
    # print(personal_income_by_state)
    return personal_income_by_state


def load_by_candidate_id(candidate_id: str, year: int) -> pd.DataFrame:
    """
    Load dataframe corresponding to candidate_id
    :param candidate_id:
    :param year:
    :return:
    """
    logger.info(f'Attempting to load data for {candidate_id=}')
    fpath = os.path.join("Votesmart", "sigs", candidate_id + ".csv")
    if not os.path.exists(fpath):
        logger.warning(f'File for {candidate_id=} in the years before {year} was not found at {fpath}')
        return None

    ratings = pd.read_csv(fpath)
    logger.success(f'Data for {candidate_id=} successfully loaded for years prior to {year} from {fpath}')
    return ratings


def get_ratings(candidate_id: str, year: int = 2050) -> np.array:
    """
    Fetches report card data for a specific candidate by id and optionally by year and formats in a standardized format
    for input to data model.
    :param year:
    :param candidate_id:
    :return:
    """
    logger.info('Processing data into standardized dataframe format')
    try:
        ratings = load_by_candidate_id(candidate_id=candidate_id, year=year)
        ratings['timespan'] = pd.to_numeric(ratings['timespan'].str[0:4])
        ratings = ratings[ratings['timespan'] <= year]
        ratings['rating'] = ((ratings.rating.str.replace(r'^[^0-9]*$', '0.5', regex=True)).astype(float) / 50) - 1
        ratings = ratings.rename(columns=lambda x: re.sub(r'^[a-zA-Z_]*name_', 'category_name_', x))
        ratings = ratings.rename(columns=lambda x: re.sub(r'^[a-zA-Z_]*id_', 'category_id_', x))

        pivot_columns = ratings.columns[9::2]
        temp = ratings.dropna(subset=['category_id_1'])
        temp.drop(f'category_id_1', axis=1, inplace=True)
        temp.drop(f'category_name_1', axis=1, inplace=True)

        for i in range(2, 2 + len(pivot_columns)):
            temp = temp.dropna(subset=[f'category_id_{i}'])
            temp = temp.to_dict('records')
            for entry in temp:
                entry['category_id_1'] = entry[f'category_id_{i}']
                entry['category_name_1'] = entry[f'category_name_{i}']
                ratings = ratings.append(entry, ignore_index=True)
            temp = pd.DataFrame(temp)
            ratings = ratings.drop(f'category_id_{i}', axis=1)
            ratings = ratings.drop(f'category_name_{i}', axis=1)

        ratings = ratings[['candidate_id', 'category_name_1', 'rating']]
        ratings = ratings.groupby(['category_name_1']).mean().T
        ratings = ratings.iloc[-1].to_dict()

        result = POSSIBLE_RATING_CATEGORIES
        for key in ratings.keys():
            result[key] = ratings[key]

        logger.success(
            f'Data processed for candidate {candidate_id} with a total of {len(ratings.keys())} valid rating categories')
        return result

    except KeyError:
        logger.error(f'KeyError exception processing {candidate_id=}')
        return False

    return None


def find_possible_categories() -> pd.DataFrame:
    """
    Parse candidate folder to determine possible voting categories
    """
    logger.info("Identifying unique voting category names")

    fpath = os.path.join("Votesmart", "sigs")
    fpaths = [os.path.join(fpath, x) for x in os.listdir(fpath)]
    categories = set([])

    for fpath in fpaths:
        try:
            ratings = pd.read_csv(fpath)
            if len(ratings.columns) > 3:
                categories = categories.union(set(ratings['category_name_1'].unique()))
        except KeyError:
            None
    logger.debug(categories)

    logger.info("Unique categories printed to terminal")
    return categories


def test():
    # get_ratings('21280')
    # print(find_possible_categories(cutoff_year=1990))

    return None


test()
