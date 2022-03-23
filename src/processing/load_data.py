import os
import re
import pandas as pd
import numpy as np
from loguru import logger
import utils

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
PERSONAL_INCOME_BY_STATE = pd.read_csv(os.path.join('data', 'SAINC1__ALL_AREAS_1929_2020.csv'))
DEMOGRAPHICS = pd.read_csv(os.path.join('data', '90sData.csv'), encoding='latin-1')
DEMOGRAPHICS = DEMOGRAPHICS.groupby(['Year of Estimate', 'FIPS State']).sum().reset_index()
#STATE_INCOME_TAX = pd.read_csv(os.path.join('data', 'TBD_STATE_INCOME_TAX.csv'))
#FEDERAL_INCOME_TAX = pd.read_csv(os.path.join('data', 'TBD_FEDERAL_INCOME_TAX.csv'))
        


def get_candidates(cutoff_year: int = 1900) -> pd.DataFrame:
    """
    Load data from data directory
    Load representative data; this data will serve as the basis for the rest of the data
    It will be processed by converting each election into a unique identifier in the form of 'year_stateName'
    This can then be used to pull data from other sources where the year and state name match
    :param cutoff_year:
    """
    house_rep = pd.read_csv(os.path.join('data', '1976_2020_house.csv'), sep=',', encoding='latin-1').dropna(subset=['candidate'])
    house_rep_cols = ["year", "state", "state_po", "state_fips", "state_cen", "state_ic", "office", "district", "stage",
                      "special", "candidate", "party", "candidatevotes", "totalvotes"]
    for col in house_rep.columns:
        if col not in house_rep_cols:
            house_rep.drop(col, axis=1, inplace=True)
    house_rep = house_rep[house_rep['year'] >= cutoff_year]
    string_dtypes = house_rep.convert_dtypes().select_dtypes("string")
    house_rep[string_dtypes.columns] = string_dtypes.apply(lambda x: x.str.lower())
    logger.success(f"House Representatives loaded")
    return house_rep


def get_personal_income(year: int, state_fips: int, verbose: bool = False) -> list:
    """
    Load personal income data from directory and return as a dataframe
    :param verbose:
    :param state_fips:
    :param year:
    """
    try:
        # personal_income_by_state = pd.read_csv(os.path.join('data', 'SAINC1__ALL_AREAS_1929_2020.csv'))
        personal_income_by_state = list(PERSONAL_INCOME_BY_STATE[PERSONAL_INCOME_BY_STATE['StateFips'] == state_fips][str(year)])
        overall_us_income = list(PERSONAL_INCOME_BY_STATE[PERSONAL_INCOME_BY_STATE['StateFips'] == 0][str(year)])
        personal_income_by_state = [20*state/us for state, us in zip(personal_income_by_state, overall_us_income)]
    except FileNotFoundError:
        if verbose:
            logger.warning(f'KeyError:\t\tException while loading income data for {state_fips=} and {year=}')
    # print(list(personal_income_by_state))
    return personal_income_by_state


def load_by_candidate_id(candidate_id: str, year: int, verbose: bool = False) -> pd.DataFrame:
    """
    Load dataframe corresponding to candidate_id
    :param verbose:
    :param candidate_id:
    :param year:
    :return:
    """
    fpath = os.path.join("Votesmart", "sigs", f'{candidate_id}.csv')
    if not os.path.exists(fpath):
        if verbose:
            logger.warning(f'File for {candidate_id=} in the years before {year} was not found at {fpath}')
        return None
    else:
        ratings = pd.read_csv(fpath)
    return ratings


def get_ratings(candidate_id: str, year: int = 2050, verbose: bool = False) -> list:
    """
    Fetches report card data for a specific candidate by id and optionally by year and formats in a standardized format
    for input to data model.
    :param verbose:
    :param year:
    :param candidate_id:
    :return:
    """
    try:
        ratings = load_by_candidate_id(candidate_id=candidate_id, year=year, verbose=verbose)
        ratings['timespan'] = pd.to_numeric(ratings['timespan'].astype(str).str[0:4])
        ratings = ratings[ratings['timespan'] <= int(year)]
        ratings['rating'] = ((ratings.rating.astype(str).str.replace(r'^[^0-9]*$', '0.5', regex=True)).astype(float) / 50) - 1
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

        result = TEMPLATE_RATING_DICT
        for key in ratings.keys():
            result[key] = ratings[key]
        result = [float(result[key]) for key in result.keys()]
        # print(result)
        return result

    except KeyError:
        if verbose:
            logger.warning(f'KeyError:\t\tException processing {candidate_id=}')
        return False

    except TypeError:
        if verbose:
            logger.warning(f'TypeError:\t\tException processing {candidate_id=}')
        return False


def get_population_data(year: int = 1990, state_fips: str = "", verbose: bool = False) -> list:
    """
    Returns demographic information for a given election by year and state fips code
    :param verbose:
    :param year:
    :param state_fips:
    :return:
    """
    try:
        demographics = DEMOGRAPHICS[DEMOGRAPHICS['Year of Estimate'] == year]
        demographics = demographics[demographics['FIPS State'] == state_fips]
        demographics = demographics.to_numpy()[0][3:]/100
        demographics2 = demographics / np.sum(demographics)
    except FileNotFoundError:
        if verbose:
            logger.warning(f'FileNotFoundError:\tException processing 90sData.csv on {year=}, for {state_fips}')
            return None
    except IndexError:
        if verbose:
            logger.warning(f'IndexError:\t\tException processing 90sData.csv on {year=}, for {state_fips}')
            return None
    return list(demographics + demographics2)


def get_winner_data(candidates: pd.DataFrame, year: int = 1995, state_fips: int = 1, verbose: bool = False) -> list:
    """
    Identifies the winning political candidate from a specified year and state and returns the candidate_ids
    :param verbose:
    :param candidates:
    :param year:
    :param state_fips:
    :return:
    """

    ids = []
    parties = []
    state_abbr = utils.get_state_abbr(fips=state_fips)
    candidates = pd.DataFrame(candidates[candidates['year'] == year])

    candidates = candidates[candidates['state_fips'] == state_fips]
    candidates = candidates.sort_values('candidatevotes', ascending=False).drop_duplicates(['year', 'state_fips', 'district'])

    if not candidates.empty:
        all_parties = list(candidates['party'])
        candidates = list(candidates['candidate'])
        candidates = [candidate.split(' ') for candidate in candidates]
        candidates = [[re.sub(r'[a-zA-Z]*[^a-zA-Z]+[a-zA-Z]*', '', name_seg) for name_seg in candidate] for candidate in
                      candidates]
        candidates = [[re.sub(r'^(ii)|(iii)|(jr)|(sr)$', '', name_seg) for name_seg in candidate] for candidate in
                      candidates]
        candidates = [[name_seg for name_seg in candidate if name_seg] for candidate in candidates]
        candidates = [candidate[0] + ' ' + candidate[-1] for candidate in candidates if candidate]

        for i, candidate in enumerate(candidates):
            try:
                candidate = candidate.split(' ')
                fpath = os.path.join('Votesmart', 'cands', f'{candidate[-1]}.csv')
                candidate_info = pd.read_csv(fpath)
                candidate_info = candidate_info[candidate_info['first_name'].str.lower() == candidate[0]]
                candidate_info = candidate_info[candidate_info['election_year'] == year]
                candidate_info = candidate_info[candidate_info['election_state_id'] == state_abbr]
                candidate_info = candidate_info[candidate_info['election_office'] == 'U.S. House']
                candidate_id = candidate_info['candidate_id']
                parties += [all_parties[i]]
                ids += list(candidate_id)
            except FileNotFoundError:
                if verbose:
                    logger.warning(f'FileNotFoundError:\t{state_abbr=}, {year=}, {candidate=}')
                    # df = pd.read_csv(os.path.join('Votesmart', 'candidates2.csv'))
                    # df = df.append({"last_name": candidate[-1]}, ignore_index=True)
                    # df.to_csv(os.path.join('Votesmart', 'candidates2.csv'))
            except IndexError:
                if verbose:
                    logger.warning(f'IndexError:\t\t{state_abbr=}, {year=}')
            except UnicodeDecodeError:
                if verbose:
                    logger.warning(f'UnicodeDecodeError:\t{state_abbr=}, {year=}')
    ids = list(ids)
    return ids, parties


def vectorize_party(possible_parties: list, cand_party: str):
    """
    Takes possible parties as an input to use One Hot encoding to vectorize party name
    :param possible_parties:
    :param cand_party:
    :return:
    """
    vectorized_parties = []
    for party in possible_parties:
        if party == cand_party:
            vectorized_parties += [1]
        else:
            vectorized_parties += [0]
    return vectorized_parties


def get_taxes(year: int, state_fips: int, verbose: bool = False) -> list: 
    """
    Load tax information from directory and return as a list
    :param verbose:
    :param state_fips:
    :param year:
    """
    if verbose:
        print("You made a mistake calling this function...")
    try:
        #TODO Format income tax data
        state_income_tax =  list(STATE_INCOME_TAX[STATE_INCOME_TAX["StateFips"] == state_fips][str(year)])
        federal_income_tax = list(FEDERAL_INCOME_TAX[FEDERAL_INCOME_TAX[str(year)]])
        combined_income_tax_info = federal_income_tax
        combined_income_tax_info.append(state_income_tax)
        #TODO Merge Federal and state into single list
    except FileNotFoundError:
        if verbose:
            logger.warning(f'KeyError:\t\tException while loading income data for {year=}')
    # print(list(combined_income_tax_info))
    return combined_income_tax_info

def get_election_district(year: int, fips_code: int, verbose: bool = False) -> int:
    # TODO
    return 0


def test():
    # get_ratings('21280')
    # print(find_possible_categories(cutoff_year=1990))

    return None


test()
