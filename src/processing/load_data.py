import os
import re
import pandas as pd
import numpy as np
from loguru import logger
import utils

pd.options.mode.chained_assignment = None

# STATE_INCOME_TAX = pd.read_csv(os.path.join('data', 'TBD_STATE_INCOME_TAX.csv'))
# FEDERAL_INCOME_TAX = pd.read_csv(os.path.join('data', 'TBD_FEDERAL_INCOME_TAX.csv'))


class LoadError(Exception):
    pass


class dataLoader:
    def __init__(self, verbose=False, cutoff_year=1900):
        logger.info(f'Initializing dataLoader with cutoff_year {cutoff_year} while {verbose=}')

        self.cutoff_year = cutoff_year
        self.verbose = verbose

        self.personal_income_by_state = pd.read_csv(os.path.join('data', 'SAINC1__ALL_AREAS_1929_2020.csv'))
        demographics = pd.read_csv(os.path.join('data', 'Demographics1990_2010.csv'), encoding='latin-1')
        self.demographics = demographics.groupby(['Year of Estimate', 'FIPS State']).sum().reset_index()
        self.candidates = self.get_candidates()
        self.FIPS_relations = pd.read_csv(os.path.join('data', 'FIPS.csv'))
        self.tax_burdens = pd.read_csv(os.path.join('data', 'tax_burden_by_state.csv'))
        self.marijuana_legalization_status = pd.read_csv(os.path.join('data', 'marijuana_legalization_status.csv'))

        self.sig_data_fpath = os.path.join('Votesmart', 'sig_agg', 'ALL_SIG_DATA.csv')
        if os.path.exists(self.sig_data_fpath):
            self.ALL_SIG_DATA = pd.read_csv(self.sig_data_fpath)
        else:
            self.ALL_SIG_DATA = pd.DataFrame(columns=['candidate_id', 'sig_id', 'sig_name', 'rating', 'rating_name',
                                                      'timespan', 'category_id', 'category_name'])
        self.possible_rating_categories = self.ALL_SIG_DATA['category_name_1'].unique()
        self.loaded_cand_ids = self.ALL_SIG_DATA['candidate_id'].unique()
        self.template_rating_dictionary = {key: 0 for key in self.possible_rating_categories}
        self.possible_parties = utils.find_possible_parties(self.candidates)

        self.missing_cand_ids = []

    def get_candidates(self) -> pd.DataFrame:
        """
        Load data from data directory
        Load representative data; this data will serve as the basis for the rest of the data
        It will be processed by converting each election into a unique identifier in the form of 'year_stateName'
        This can then be used to pull data from other sources where the year and state name match
        :param cutoff_year:
        """
        house_rep = pd.read_csv(os.path.join('data', '1976_2020_house.csv'), sep=',', encoding='latin-1').dropna(
            subset=['candidate'])
        house_rep_cols = ["year", "state", "state_po", "state_fips", "state_cen", "state_ic", "office", "district",
                          "stage",
                          "special", "candidate", "party", "candidatevotes", "totalvotes"]
        for col in house_rep.columns:
            if col not in house_rep_cols:
                house_rep.drop(col, axis=1, inplace=True)
        house_rep = house_rep[house_rep['year'] >= self.cutoff_year]
        string_dtypes = house_rep.convert_dtypes().select_dtypes("string")
        house_rep[string_dtypes.columns] = string_dtypes.apply(lambda x: x.str.lower())
        logger.success(f"House Representatives loaded")
        return house_rep

    def get_personal_income(self, year: int, state_fips: int, verbose: bool = False) -> list:
        """
        Load personal income data from directory and return as a dataframe
        :param verbose:
        :param state_fips:
        :param year:
        """
        PERSONAL_INCOME_BY_STATE = self.personal_income_by_state
        try:
            # personal_income_by_state = pd.read_csv(os.path.join('data', 'SAINC1__ALL_AREAS_1929_2020.csv'))
            personal_income_by_state = list(
                PERSONAL_INCOME_BY_STATE[PERSONAL_INCOME_BY_STATE['StateFips'] == state_fips][str(year)])
            # overall_us_income = list(PERSONAL_INCOME_BY_STATE[PERSONAL_INCOME_BY_STATE['StateFips'] == 0][str(year)])
            # personal_income_by_state = [20 * state / us for state, us in
            #                             zip(personal_income_by_state, overall_us_income)]
        except FileNotFoundError:
            raise LoadError(f'FileNotFoundError:\t\tException while loading income data for {state_fips=} and {year=} i'
                            f'n load_data.get_personal_income()')
        # print(list(personal_income_by_state))
        return personal_income_by_state

    def load_by_candidate_id(self, candidate_id: str, year: int, verbose: bool = False) -> pd.DataFrame:
        """
        Load dataframe corresponding to candidate_id
        :param verbose:
        :param candidate_id:
        :param year:
        :return:
        """
        fpath = os.path.join("Votesmart", "sigs", f'{candidate_id}.csv')
        if not os.path.exists(fpath):
            self.missing_cand_ids += [candidate_id]
            raise LoadError(f'File for {candidate_id=} in the years before {year} was not found at {fpath} in load_data'
                            f'.load_by_candidate_id()')
        else:
            ratings = pd.read_csv(fpath)
        return ratings

    def process_ratings(self, ratings: pd.DataFrame, candidate_id: str, year: int):
        """
        Filters ratings dataframe
        :param year:
        :param candidate_id:
        :param ratings:
        """
        # print(candidate_id, year)
        ratings = ratings[ratings['candidate_id'] == candidate_id]
        ratings['timespan'] = pd.to_numeric(ratings['timespan'])
        ratings = ratings[pd.to_numeric(ratings['timespan']) <= year]

        if not ratings.empty:

            ratings = ratings[['candidate_id', 'category_name_1', 'rating']]
            ratings = ratings.groupby(['category_name_1']).mean()['rating'].T.to_dict()

            temp = self.template_rating_dictionary
            for category in ratings.keys():
                temp[category] = ratings[category]

            result = []
            for key, val in temp.items():
                result += [val]
            return result
        else:
            return []

    def get_ratings(self, candidate_id: str, year: int = 2050, verbose: bool = False) -> list:
        """
        Fetches report card data for a specific candidate by id and optionally by year and formats in a standardized format
        for input to data model.
        :param verbose:
        :param year:
        :param candidate_id:
        :return:
        """

        # ratings = self.process_ratings(self.ALL_SIG_DATA, candidate_id=candidate_id, year=year)

        if candidate_id not in self.loaded_cand_ids:
            # logger.info('Getting Ratings from .csv')
            try:
                ratings = self.load_by_candidate_id(candidate_id=candidate_id, year=year, verbose=verbose)
                ratings['timespan'] = pd.to_numeric(ratings['timespan'].astype(str).str[0:4])
                ratings['rating'] = ((ratings.rating.astype(str).str.replace(r'^[^0-9]*$', '0.5', regex=True)).astype(
                    float) / 50) - 1
                ratings = ratings.rename(columns=lambda x: re.sub(r'^[a-zA-Z_]*name_', 'category_name_', x))
                ratings = ratings.rename(columns=lambda x: re.sub(r'^[a-zA-Z_]*id_', 'category_id_', x))

                pivot_columns = ratings.columns[9:]
                ratings = ratings.dropna(subset=['category_id_1', 'category_name_1'])
                temp = ratings

                for i in range(2, 2 + int(len(pivot_columns) / 2)):
                    temp = temp.dropna(subset=[f'category_id_{i}', f'category_name_{i}'])
                    temp['category_name_1'] = temp[f'category_name_{i}']
                    temp['category_id_1'] = temp[f'category_id_{i}']
                    add_to_ratings = temp.drop(pivot_columns, axis=1).to_dict('records')
                    for entry in add_to_ratings:
                        ratings = ratings.append(entry, ignore_index=True)

                ratings = ratings.drop(pivot_columns, axis=1)
                if 'category_id' in ratings.columns:
                    ratings = ratings.drop(['category_id'], axis=1)
                if 'category_name' in ratings.columns:
                    ratings = ratings.drop(['category_name'], axis=1)

                self.ALL_SIG_DATA = pd.concat([self.ALL_SIG_DATA, ratings])
                self.save_processed_cand_data()
                self.loaded_cand_ids = self.ALL_SIG_DATA['candidate_id'].unique()

            except KeyError:
                raise LoadError(f'KeyError:\t\tException processing {candidate_id=} in load_data.get_ratings()')

            except TypeError:
                raise LoadError(f'TypeError:\t\tException processing {candidate_id=} in load_data.get_ratings()')

        result = self.process_ratings(self.ALL_SIG_DATA, candidate_id=candidate_id, year=year)
        return result

    def get_population_data(self, year: int = 1990, state_fips: str = "", verbose: bool = False) -> list:
        """
        Returns demographic information for a given election by year and state fips code
        :param verbose:
        :param year:
        :param state_fips:
        :return:
        """
        try:
            demographics = self.demographics[self.demographics['Year of Estimate'] == year]
            demographics = demographics[demographics['FIPS State'] == state_fips]
            demographics = demographics.to_numpy()[0][3:] / 100
            demographics2 = demographics / np.sum(demographics)
        except FileNotFoundError:
            raise LoadError(f'FileNotFoundError:\tException processing demographics on {year=}, for {state_fips} in loa'
                            f'd_data.get_population_data()')
        except IndexError:
            raise LoadError(f'IndexError:\t\tException processing demographics on {year=}, for {state_fips} in load_dat'
                            f'a.get_population_data()')
        return list(demographics + demographics2)

    def get_winner_data(self, year: int = 1995, state_fips: int = 1, verbose: bool = False) -> list:
        """
        Identifies the winning political candidate from a specified year and state and returns the candidate_ids
        :param average:
        :param verbose:
        :param candidates:
        :param year:
        :param state_fips:
        :return:
        """

        ids = []
        parties = []
        state_abbr = utils.get_state_abbr(self.FIPS_relations, fips=state_fips)
        candidates = pd.DataFrame(self.candidates[self.candidates['year'] == year])

        candidates = candidates[candidates['state_fips'] == state_fips]
        candidates = candidates.sort_values('candidatevotes', ascending=False).drop_duplicates(
            ['year', 'state_fips', 'district'])

        if not candidates.empty:
            all_parties = list(candidates['party'])
            candidates = list(candidates['candidate'])
            candidates = [candidate.split(' ') for candidate in candidates]
            candidates = [[re.sub(r'[a-zA-Z]*[^a-zA-Z]+[a-zA-Z]*', '', name_seg) for name_seg in candidate] for
                          candidate in
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

    def vectorize_party(self, cand_party: str):
        """
        Takes possible parties as an input to use One Hot encoding to vectorize party name
        :param possible_parties:
        :param cand_party:
        :return:
        """
        vectorized_parties = []
        for party in self.possible_parties:
            if party == cand_party:
                vectorized_parties += [1]
            else:
                vectorized_parties += [0]
        return vectorized_parties

    def get_taxes(self, year: int, state_fips: int, verbose: bool = False) -> list:
        """
        Load tax information from directory and return as a list
        :param verbose:
        :param state_fips:
        :param year:
        """
        if verbose:
            print("You made a mistake calling this function...")
        try:
            # TODO Format income tax data
            state_income_tax = list(STATE_INCOME_TAX[STATE_INCOME_TAX["StateFips"] == state_fips][str(year)])
            federal_income_tax = list(FEDERAL_INCOME_TAX[FEDERAL_INCOME_TAX[str(year)]])
            combined_income_tax_info = federal_income_tax
            combined_income_tax_info.append(state_income_tax)
            # TODO Merge Federal and state into single list
        except FileNotFoundError:
            if verbose:
                logger.warning(f'KeyError:\t\tException while loading income data for {year=}')
        # print(list(combined_income_tax_info))
        return combined_income_tax_info

    def save_processed_cand_data(self):
        missing_cand_ids = list(set(self.missing_cand_ids))
        missing_cand_ids = pd.DataFrame(missing_cand_ids, columns=['cand_id'])
        missing_cand_ids.to_csv(os.path.join('Votesmart', 'cand_ids.csv'), index=False)

        self.ALL_SIG_DATA.to_csv(self.sig_data_fpath, index=False)
        logger.success('Sig Data file generated, run Votesmart script to generate name data')

    def get_tax_burden_data(self, year: int, state_fips: int, verbose: bool = False) -> list:
        """
        Finds tax burden data for given year and state
        :param year:
        :param state_fips:
        :param verbose:
        :return:
        """
        state_name = utils.get_state_name(self.FIPS_relations, fips=state_fips)
        tax_burden = self.tax_burdens[self.tax_burdens['State'] == state_name]
        tax_burden = tax_burden[tax_burden['Year'] == year]
        tax_burden = tax_burden.drop(columns=['Year', 'State']).values.tolist()[0]

        return tax_burden

    def get_marijuana_legalization_status(self, year: int, state_fips: int, verbose: bool = False) -> list:
        """
        Returns vector detailing if state has legalized marijuana during a certain year
        :param year:
        :param state_fips:
        :param verbose:
        :return:
        """
        state_name = utils.get_state_name(self.FIPS_relations, fips=state_fips)
        leg_status = self.marijuana_legalization_status[self.marijuana_legalization_status['State'] == state_name]
        result = [-1, -1]
        if (leg_status['Recreational'] <= year).any():
            result[0] = 1
        if (leg_status['Medicinal'] <= year).any():
            result[1] = 1

        return result

