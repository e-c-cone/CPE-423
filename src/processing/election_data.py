import os
import time
import json
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utils
import processing.load_data as load_data


def generate_dataset(cutoff_year: int = 1990, verbose: bool = False, reload_data: bool = False):
    """
    Generates a dataset for use in training as part of overall data pipeline.
    :param reload_data:
    :param cutoff_year:
    :param verbose:
    :return:
    """
    processed_fpath = os.path.join("data", "processed_dataset.csv")

    if reload_data or not os.path.exists(processed_fpath):
        if not reload_data and not os.path.exists(processed_fpath):
            logger.warning(f'File was not found at {processed_fpath}, but reload_data was not passed as an argument. Ru'
                           f'n code with -rd or --reload_data to reload data. Reloading data despite missing argument.')
        logger.info('Generating viable dataset')

        FIPS = pd.read_csv(os.path.join('data', 'FIPS.csv'))
        FIPS_POSSIBLE = FIPS['fips'].to_numpy()
        house_rep = load_data.get_candidates(cutoff_year=cutoff_year)

        # Load winning ids into memory in order to retrieve ratings
        reps = []
        data = {}
        for state in FIPS_POSSIBLE:
            for year in range(cutoff_year, 2000)[::2]:  # TODO - make this range longer when we have more data
                winner_ids, parties = \
                    load_data.get_winner_data(candidates=house_rep, year=year, state_fips=state, verbose=verbose)
                for i, id in enumerate(winner_ids):
                    try:
                        data[f'{state}_{year}_{i}'] = {"id": id, "party": parties[i]}
                        if id:
                            reps += [id]
                    except IndexError:
                        logger.warning(f'IndexError for {id=}\tat index={i}')
                        print(winner_ids, parties)
        logger.success(f'There are {len(set(reps))} valid loaded representatives')

        # Load data from elections and merge into one dataset
        logger.info('Merging datasets')
        X = []
        Y = []
        data_size = len(data.keys())
        start = time.time()
        POSSIBLE_PARTIES = utils.find_possible_parties(house_rep)
        for i, election_winner in enumerate(data.keys()):

            # Print information on estimated time remaining for data processing
            if verbose and (i % int(data_size/20) == 0 or i == 10) and i > 0:
                dur = time.time() - start
                remaining = int(dur * (data_size - i) / i)
                remaining = time.strftime("%H:%M:%S", time.gmtime(remaining))
                logger.debug(f'Merging datasets... election winner {i} out of'
                             f' {data_size}: estimated time {remaining=}')
            elif not verbose and (i % int(data_size/3) == 0 or i == 10) and i > 0:
                logger.info(f'A total of {i}/{data_size} data points have been merged')

            # Load data for each election and add to running dataset
            state_fips = int(election_winner.split('_')[0])
            year = int(election_winner.split('_')[1])
            pop = load_data.get_population_data(year=year, state_fips=state_fips, verbose=verbose)
            inc = load_data.get_personal_income(year=year, state_fips=state_fips, verbose=verbose)
            party = load_data.vectorize_party(POSSIBLE_PARTIES, data[election_winner]['party'])
            rating = load_data.get_ratings(data[election_winner]['id'], int(election_winner.split('_')[1]), verbose=verbose)
            # print(rating)
            if pop and inc and party and rating:
                X += [[year-1990] + pop + inc + party]
                Y += [rating]
            # if not rating:
            #     logger.warning(f'FUCK THE RATING ISNT WORKING {i=}, {election_winner=}')

        # Perform PCA on input data to reduce dimensionality
        sc = StandardScaler()
        pca1 = PCA(n_components='mle')
        # pca2 = PCA(n_components='mle')
        if verbose:
            logger.info(f'Before Processing: {np.array(X).shape=}, {np.array(Y).shape=}')
        X = sc.fit_transform(X)
        X = pca1.fit_transform(X)
        # Y = pca2.fit_transform(Y)
        if verbose:
            logger.info(f'After Processing:  {np.array(X).shape=}, {np.array(Y).shape=}')

        # Save data as .json
        dataset = {"X": np.array(X).tolist(), "Y": np.array(Y).tolist()}
        with open(processed_fpath, 'w') as file:
            json.dump(dataset, file, indent=4)
            if verbose:
                logger.success(f'File saved successfully to {processed_fpath}')

        logger.success(f'Loaded {len(X)} data samples into memory')
        return X, Y
    else:
        logger.info('Loading dataset from file')

        # Retrieve data from file
        # dataset = pd.read_csv(processed_fpath)
        with open(processed_fpath, 'r') as file:
            dataset = json.load(file)
        X = dataset["X"]
        Y = dataset["Y"]

        logger.success(f'Data was loaded successfully from {processed_fpath}.')
        return X, Y


