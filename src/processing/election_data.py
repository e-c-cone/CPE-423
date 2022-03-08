import os
import time
import json
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
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
            for year in range(cutoff_year, 2000)[::2]:  # TODO
                winner_ids = load_data.get_winner_ids(candidates=house_rep, year=year, state_fips=state, verbose=verbose)
                data[f'{state}_{year}'] = winner_ids
                if winner_ids:
                    reps += winner_ids
        logger.success(f'There are {len(set(reps))} valid loaded representatives')

        # Load data from elections and merge into one dataset
        logger.info('Merging datasets')
        X = []
        Y = []
        data_size = len(data.keys())
        start = time.time()
        for i, election in enumerate(data.keys()):

            # Print information on estimated time remaining for data processing
            if verbose and (i % int(data_size/20) == 0 or i == 10) and i > 0:
                dur = time.time() - start
                remaining = int(dur * (data_size - i) / i)
                remaining = time.strftime("%H:%M:%S", time.gmtime(remaining))
                logger.debug(f'Merging datasets... election {i} out of {data_size} with {len(data[election])} candidate'
                             f's:\testimated time {remaining=}')
            elif not verbose and (i % int(data_size/3) == 0 or i == 10) and i > 0:
                logger.info(f'A total of {i}/{data_size} data points have been merged')

            # Load data for each election and add to running dataset
            for j, candidate_id in enumerate(data[election]):
                if j <= 1000:  # TODO
                    state_fips = int(election.split('_')[0])
                    year = int(election.split('_')[1])
                    pop = load_data.get_population_data(year=year, state_fips=state_fips, verbose=verbose)
                    inc = load_data.get_personal_income(year=year, state_fips=state_fips, verbose=verbose)
                    rating = load_data.get_ratings(candidate_id, election.split('_')[1])
                    if pop and inc and rating:
                        X += [pop + inc]
                        Y += [rating]

        # Save data as .json
        dataset = {"X": X, "Y": Y}
        # dataset.to_csv(processed_fpath)
        with open(processed_fpath, 'w') as file:
            json.dump(dataset, file, indent=4)
            if verbose:
                logger.success(f'File saved successfully to {processed_fpath}')

        logger.success(f'Loaded {len(X)} data samples into memory')
        if verbose:
            logger.info(f'{np.array(X).shape=}, {np.array(Y).shape=}')
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


