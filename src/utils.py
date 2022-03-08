import os
import pandas as pd
from loguru import logger


def get_fips(abbreviation: str = "", state: str = "") -> int:
    """
    Finds the state FIPs code when given a state abbreviation or name
    :param abbreviation:
    :param state:
    :return:
    """
    relationship_table = pd.read_csv(os.path.join('data', 'FIPS.csv'))
    if abbreviation:
        code = relationship_table[relationship_table['abbreviation'] == abbreviation]
        code = code['fips']
        return list(code)[0]
    elif state:
        code = relationship_table[relationship_table['state'] == state]
        code = code['fips']
        return list(code)[0]
    return 0


def get_state_name(abbreviation: str = "", fips: int = 0) -> str:
    """
    Finds the state name when given a state abbreviation or fips code
    :param fips:
    :param abbreviation:
    :return:
    """
    relationship_table = pd.read_csv(os.path.join('data', 'FIPS.csv'))
    if abbreviation:
        name = relationship_table[relationship_table['abbreviation'] == abbreviation]
        name = name['name']
        return list(name)[0]
    elif fips:
        name = relationship_table[relationship_table['fips'] == fips]
        name = name['name']
        return list(name)[0]
    return ""


def get_state_abbr(name: str = "", fips: int = 0) -> str:
    """
    Finds the state name when given a state abbreviation or fips code
    :param fips:
    :param name:
    :return:
    """
    relationship_table = pd.read_csv(os.path.join('data', 'FIPS.csv'))
    if name:
        abbr = relationship_table[relationship_table['name'] == name]
        abbr = abbr['abbreviation']
        return list(abbr)[0]
    elif fips:
        abbr = relationship_table[relationship_table['fips'] == fips]
        abbr = abbr['abbreviation']
        return list(abbr)[0]
    return ""


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


def find_possible_parties(candidates: pd.DataFrame) -> list:
    """
    Find a list of all possible parties
    :param candidates:
    :return:
    """
    return list(candidates['party'].unique())

