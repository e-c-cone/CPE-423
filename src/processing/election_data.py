import os
import time
import json
import random
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utils
from processing.load_data import dataLoader, LoadError


class data_generator:
    def __init__(self, cutoff_year: int = 2012, verbose: bool = False, reload_data: bool = False):
        self.cutoff_year = cutoff_year
        self.verbose = verbose
        self.reload_data = reload_data
        self.load_data = dataLoader(verbose=verbose, cutoff_year=cutoff_year)

        FIPS = pd.read_csv(os.path.join('data', 'FIPS.csv'))
        self.FIPS_POSSIBLE = FIPS['fips'].to_numpy()

    def generate_dataset(self):
        """
        Generates a dataset for use in training as part of overall data pipeline.
        :param reload_data:
        :param cutoff_year:
        :param verbose:
        :return:
        """
        cutoff_year = self.cutoff_year
        verbose = self.verbose
        reload_data = self.reload_data
        processed_fpath = os.path.join("data", "processed_dataset.csv")

        if reload_data or not os.path.exists(processed_fpath):
            if not reload_data and not os.path.exists(processed_fpath):
                logger.warning(f'File was not found at {processed_fpath}, but reload_data was not passed as an argument'
                               f'. Run code with -rd or --reload_data to reload data. Reloading data despite missing'
                               f'argument.')
            logger.info('Generating viable dataset')

            # Load winning ids into memory in order to retrieve ratings
            reps = []
            data = {}
            for state in self.FIPS_POSSIBLE:
                for year in range(1990, cutoff_year+2)[::2]:  # TODO - get data after 2010
                    winner_ids, winner_additional_data = \
                        self.load_data.get_winner_data(year=year, state_fips=state, verbose=verbose)
                    second_best_ids, loser_additional_data = \
                        self.load_data.get_winner_data(year=year, state_fips=state, verbose=verbose, second_best=True)

                    if len(winner_additional_data['percent_votes']) < len(winner_ids):
                        winner_ids = winner_ids[:len(winner_additional_data['percent_votes'])]

                    for i, winner_id in enumerate(winner_ids):
                        if winner_id and type(winner_id) == int:
                            if i < len(second_best_ids):
                                second_best_id = second_best_ids[i]
                            else:
                                second_best_id = winner_id
                                try:
                                    loser_additional_data['parties'] += list(winner_additional_data['parties'])[i]
                                    loser_additional_data['percent_votes'] += list(winner_additional_data['percent_votes'])[i]
                                except TypeError:
                                    print(loser_additional_data)
                                    print(winner_additional_data)
                                    continue
                                except IndexError:
                                    print(loser_additional_data)
                                    print(winner_additional_data)
                                    continue
                            if type(second_best_id) == str and "Error" in second_best_id:
                                second_best_id = winner_id
                            reps += [winner_id, second_best_id]
                            data[f'{state}_{year}_{i}'] = {"winner": winner_id, "second_best": second_best_id,
                                                           "winner_perc": winner_additional_data['percent_votes'][i],
                                                           "loser_perc": loser_additional_data['percent_votes'][i]}

            logger.success(f'There are {len(set(reps))} valid loaded representatives')
            pd.DataFrame.from_dict(data).T.to_csv('TEST.csv')

            # Load data from elections and merge into one dataset
            logger.info('Merging datasets')
            x_categorical = []
            x_quantitative = []
            Y = []
            additional_data = {"election_id": [], "second_best_ratings": [], "winner_perc": [], "loser_perc": []}
            successful_keys = []
            prev_err = None
            data_size = len(data.keys())
            start = time.time()
            for i, election_id in enumerate(data.keys()):

                # Print information on estimated time remaining for data processing
                if verbose and (i % int(data_size / 20) == 0 or i == 10) and i > 0:
                    dur = time.time() - start
                    remaining = int(dur * (data_size - i) / i)
                    remaining = time.strftime("%H:%M:%S", time.gmtime(remaining))
                    logger.debug(f'Merging datasets... election winner {i} out of'
                                 f' {data_size}: estimated time {remaining=}')
                elif not verbose and (i % int(data_size / 3) == 0 or i == 10) and i > 0:
                    logger.info(f'A total of {i}/{data_size} data points have been merged')

                # Load data for each election and add to running dataset
                state_fips = int(election_id.split('_')[0])
                year = int(election_id.split('_')[1])

                try:
                    pop = self.load_data.get_population_data(year=year, state_fips=state_fips, verbose=verbose)
                    inc = self.load_data.get_personal_income(year=year, state_fips=state_fips, verbose=verbose)
                    tax_burden = self.load_data.get_tax_burden_data(year=year, state_fips=state_fips, verbose=verbose)
                    rating = self.load_data.get_ratings(data[election_id]['winner'], year, verbose=False)
                    second_best_ratings = self.load_data.get_ratings(data[election_id]['second_best'], year,
                                                                    verbose=verbose)
                    marijuana_status = self.load_data.get_marijuana_legalization_status(year=year,
                                                                                        state_fips=state_fips,
                                                                                        verbose=verbose)

                    previous_election_ratings = []
                    if not year % 10 == 0:
                        district = election_id.split('_')[2]
                        prev_election = str(state_fips) + '_' + str(year - 2) + '_' + district
                        previous_election_ratings = self.load_data.get_ratings(data[prev_election]['winner'], year - 2,
                                                                               verbose=verbose)
                    else:
                        all_ratings = []
                        for dist in range(60):
                            prev_election = str(state_fips) + '_' + str(year - 2) + '_' + str(dist)
                            if prev_election in data:
                                prev_election_rating = self.load_data.get_ratings(data[prev_election]['winner'],
                                                                                  year - 2, verbose=verbose)
                                if np.array(prev_election_rating).shape[0] == 72:
                                    all_ratings += [prev_election_rating]

                        if np.array(all_ratings).shape[0] == 72:
                            previous_election_ratings = all_ratings
                        elif len(np.array(all_ratings).shape) > 1:
                            try:
                                previous_election_ratings = np.average(all_ratings, axis=0).tolist()
                            except ZeroDivisionError:
                                previous_election_ratings = all_ratings

                    if not np.array(previous_election_ratings).shape[0] == 72:
                        previous_election_ratings = np.zeros(72).tolist()

                    if pop and inc and tax_burden and rating and previous_election_ratings and second_best_ratings:
                        x_quantitative += [pop + inc + tax_burden + previous_election_ratings]
                        x_categorical += [[year-1990] + [state_fips] + marijuana_status]
                        Y += [rating]
                        additional_data['election_id'] += [election_id]
                        additional_data['second_best_ratings'] += [second_best_ratings]
                        additional_data['winner_perc'] += [data[election_id]['winner_perc']]
                        additional_data['loser_perc'] += [data[election_id]['loser_perc']]
                        successful_keys += [election_id]

                except LoadError as e:
                    if verbose and not e == prev_err:
                        logger.warning(e)
                        prev_err = e
                except KeyError:
                    pass

            self.load_data.save_processed_cand_data()

            # Perform PCA on input data to reduce dimensionality
            sc = StandardScaler()
            pca1 = PCA(n_components='mle')
            # pca2 = PCA(n_components='mle')
            if verbose:
                logger.info(f'Before Processing: {np.array(x_quantitative).shape=}, {np.array(Y).shape=}')
            x_quantitative = sc.fit_transform(x_quantitative)
            x_quantitative = pca1.fit_transform(x_quantitative)
            # Y = pca2.fit_transform(Y)
            if verbose:
                logger.info(f'After Processing:  {np.array(x_quantitative).shape=}, {np.array(Y).shape=}')

            X = np.array(np.concatenate((x_quantitative, x_categorical), axis=1))
            # X = np.multiply(X, np.random.normal(1, 0.05, X))
            if verbose:
                logger.info(f'After Processing:  {X.shape=}')

            # Save data as .json
            # dataset = {"X": X.tolist(), "Y": np.array(Y).tolist(),
            #            "second_best_ratings": np.array(second_best_ratings).tolist(), "keys": successful_keys}
            dataset = {"X": X.tolist(), "Y": np.array(Y).tolist(),
                       "additional_data": additional_data, "keys": successful_keys}
            with open(processed_fpath, 'w') as file:
                json.dump(dataset, file, indent=4)
                if verbose:
                    logger.success(f'File saved successfully to {processed_fpath}')

            logger.success(f'Loaded {len(X)} data samples into memory')

            return X, np.array(Y), additional_data, successful_keys
        else:
            logger.info('Loading dataset from file')

            # Retrieve data from file
            # dataset = pd.read_csv(processed_fpath)
            with open(processed_fpath, 'r') as file:
                dataset = json.load(file)
            X = dataset["X"]
            Y = dataset["Y"]
            additional_data = dataset["additional_data"]
            keys = dataset["keys"]

            logger.success(f'Data was loaded successfully from {processed_fpath}.')
            return X, Y, additional_data, keys
            
    def election_squish(self):
        df = pd.read_csv('pred_results_for_last_year.csv')
        
        newdf = pd.DataFrame()
        newdf['Y'] = [[i, j, False] if random.uniform(0, 1) > .5 else [j, i, True]
                      for i, j in zip(df['winner_perc'], df['loser_perc'])]
        
        ind = ['x_test','prediction','y_test','second_best_ratings']
        
        X = []
        for i in range(len(df['x_test'])):
            hold = []
            for k in ind:
                hold += [float(j) for j in df[k][i].strip('][').split(', ')]
            X += [hold]
            
        if not newdf['Y'][i][2]:
            newdf['X'] = pd.Series(X)
        else:
            newdf['X'] = pd.Series(X)
        
        print(len(newdf['X'][0]))
        
        for i in range(len(newdf['Y'])):
            newdf['Y'][i] = newdf['Y'][i][:-1]  # remove true/false
            if newdf['Y'][i][0] == .5:
                newdf = newdf.drop([i])     # drop rows with .5
        newdf = newdf.set_index(pd.Index(range(len(newdf['Y']))))   # update indexes
        newdf = newdf.iloc[:, [1, 0]]   # make columns [X,Y]
                
        newdf['Y_BC'] = np.ones(len(newdf['Y']))
        for i in range(len(newdf['Y_BC'])):
            if newdf['Y'][i][0] > newdf['Y'][i][1]:  
                newdf['Y_BC'][i] = 1
            else:
                newdf['Y_BC'][i] = 0
            
        return newdf
        