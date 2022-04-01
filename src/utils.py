import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.cluster import KMeans

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
                              'Economy and Fiscal', 'Higher Education', 'K-12 Education', 'National Security'}


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
            logger.warning(f'TypeError loading possible parties')
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


def generate_sig_data_file():
    """
    Goes through all processed sig files and clusters interest groups together
    :return:
    """
    sig_data_fpath = os.path.join('Votesmart', 'sig_agg', 'SIG_ALL_DATA.csv')

    if not os.path.exists(sig_data_fpath):
        sig = None
        for fname in os.listdir(os.path.join('Votesmart', 'sigs')):
            if '_p' in fname:
                sig = pd.concat([sig, pd.read_csv(os.path.join('Votesmart', 'sigs', fname))])

        sig.to_csv(sig_data_fpath)
        logger.warning('Sig Data file generated, run Votesmart script to generate name data')
        pd.DataFrame(columns=POSSIBLE_RATING_CATEGORIES, index=sig['sig_id'].unique()).to_csv(os.path.join('Votesmart', 'sig_agg', 'SIG_FAVOR_TYPE.csv'))
    # elif os.path.exists(data_fpath):
    #     data = pd.read_csv(data_fpath)
        # data.T.to_csv(data_fpath)


def compare_prediction_to_actual(predy, actualy, fname: str = 'data'):
    plt.bar([i for i in range(len(actualy))], actualy)
    plt.scatter([i for i in range(len(predy))], predy)
    plt.plot([i for i in range(len(predy))], abs(actualy-predy))
    plt.legend(loc='upper right')
    plt.savefig(os.path.join('plots', f'{fname}.png'))
    plt.clf()
