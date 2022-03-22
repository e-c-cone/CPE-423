import os
import re
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
    result = []
    candidates = list(candidates['party'].unique())
    for cand in candidates:
        try:
            if cand:
                result += [cand]
        except TypeError:
            logger.info(f'TypeError loading possible parties')
    return result


def get_proper_names(candidates: pd.DataFrame):
    """
    Extracts proper names
    :param candidates:
    :return:
    """

    # TODO - Implement better accented character handling
    logger.info(f'Beginning proper name processing')

    candidates = candidates['candidate'].tolist()
    candidates = [candidate.split(' ') for candidate in candidates]
    candidates = [[re.sub(r'[a-zA-Z]*[^a-zA-Z]+[a-zA-Z]*', '', name_seg) for name_seg in candidate] for candidate in
                  candidates]
    candidates = [[re.sub(r'^(ii)|(iii)|(jr)|(sr)$', '', name_seg) for name_seg in candidate] for candidate in
                  candidates]
    candidates = [[name_seg for name_seg in candidate if name_seg] for candidate in candidates]
    lnames = [candidate[-1] for candidate in candidates if candidate]
    lnames = pd.DataFrame({"last_name": lnames}).drop_duplicates(subset=['last_name'])
    lnames.to_csv(os.path.join('Votesmart', 'candidates2.csv'))
    candidates = [candidate[0] + ' ' + candidate[-1] for candidate in candidates if candidate]

    logger.success(f'Loaded names successfully')
    return candidates


def generate_ids_from_cand_dir():
    files = os.listdir(os.path.join("Votesmart", "cands"))
    files = [os.path.join("Votesmart", "cands", fpath) for fpath in files]

    cand_ids = []
    for fpath in files:
        try:
            tmp = pd.read_csv(fpath)
            cand_ids.extend(tmp["candidate_id"].to_list())
        except:
            print(f'Error loading data from {fpath}')

    print(len(cand_ids))
    df = pd.DataFrame(cand_ids, columns=['cand_id']).drop_duplicates(subset=['cand_id'])
    print(len(df))
    print(df.head())
    df.to_csv('cand_ids.csv')
