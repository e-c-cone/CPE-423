import os
import re
import pandas as pd
import numpy as np
from loguru import logger


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
        ratings['rating'] = (ratings.rating.str.replace(r'^[^0-9]*$', '0.5', regex=True)).astype(float)/100
        ratings = ratings.rename(columns=lambda x: re.sub(r'^[a-zA-Z_]*name_', 'category_name_', x))
        ratings = ratings.rename(columns=lambda x: re.sub(r'^[a-zA-Z_]*id_', 'category_id_', x))

        pivot_columns = ratings.columns[9::2]
        temp = ratings.dropna(subset=['category_id_1'])
        temp.drop(f'category_id_1', axis=1, inplace=True)
        temp.drop(f'category_name_1', axis=1, inplace=True)

        for i in range(2, 2+len(pivot_columns)):
            temp = temp.dropna(subset=[f'category_id_{i}'])
            temp = temp.to_dict('records')
            for entry in temp:
                entry['category_id_1'] = entry[f'category_id_{i}']
                entry['category_name_1'] = entry[f'category_name_{i}']
                ratings = ratings.append(entry, ignore_index=True)
            temp = pd.DataFrame(temp)
            ratings = ratings.drop(f'category_id_{i}', axis=1)
            ratings = ratings.drop(f'category_name_{i}', axis=1)

        result = ratings[['candidate_id', 'category_name_1', 'rating']]
        result = result.groupby(['category_name_1']).mean().T
        result = result.iloc[-1].to_dict()

        logger.success(f'Data processed for candidate {candidate_id} with a total of {len(result.keys())} valid rating categories')
        return result

    except KeyError:
        logger.error(f'KeyError exception processing {candidate_id=}')
        return False

    return None



def test():
    get_ratings('21280')


test()