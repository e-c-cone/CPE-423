import os
import time
import json
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utils
from processing.load_data import dataLoader, LoadError


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

        load_data = dataLoader(verbose=verbose, cutoff_year=cutoff_year)

        FIPS = pd.read_csv(os.path.join('data', 'FIPS.csv'))
        FIPS_POSSIBLE = FIPS['fips'].to_numpy()

        # Load winning ids into memory in order to retrieve ratings
        reps = []
        data = {}
        for state in FIPS_POSSIBLE:
            for year in range(cutoff_year, 2010)[::2]:  # TODO - make this range longer when we have more data
                winner_ids, parties = \
                    load_data.get_winner_data(year=year, state_fips=state, verbose=verbose)
                for i, id in enumerate(winner_ids):
                    try:
                        data[f'{state}_{year}_{i}'] = {"id": id, "party": parties[i]}
                        if id:
                            reps += [id]
                    except IndexError:
                        logger.warning(f'IndexError for {id=}\tat index={i}')
                        # print(winner_ids, parties)
        logger.success(f'There are {len(set(reps))} valid loaded representatives')

        # Load data from elections and merge into one dataset
        logger.info('Merging datasets')
        x_categorical = []
        x_quantitative = []
        Y = []
        prev_err = None
        data_size = len(data.keys())
        start = time.time()
        for i, election_winner in enumerate(data.keys()):

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
            state_fips = int(election_winner.split('_')[0])
            year = int(election_winner.split('_')[1])
            district = int(election_winner.split('_')[2])
            prev_election = str(state_fips) + '_' + str(year-2) + '_' + str(district)

            try:
                pop = load_data.get_population_data(year=year, state_fips=state_fips, verbose=verbose)
                inc = load_data.get_personal_income(year=year, state_fips=state_fips, verbose=verbose)
                tax_burden = load_data.get_tax_burden_data(year=year, state_fips=state_fips, verbose=verbose)
                rating = load_data.get_ratings(data[election_winner]['id'], int(election_winner.split('_')[1]),
                                               verbose=verbose)
                marijuana_status = load_data.get_marijuana_legalization_status(year=year, state_fips=state_fips,
                                                                               verbose=verbose)

                previous_election_ratings = load_data.get_ratings(data[prev_election]['id'],
                                                                  year-2, verbose=verbose)

                if pop and inc and tax_burden and rating and previous_election_ratings:
                    x_quantitative += [pop + inc + tax_burden + previous_election_ratings]
                    # randomized_x = np.multiply(quant_data, np.random.normal(1, 0.05, np.array(quant_data).shape)).tolist()
                    # print(np.array(randomized_x).shape)
                    x_categorical += [[year - 2000] + [state_fips] + marijuana_status]
                    Y += [rating]
            except LoadError as e:
                if verbose and not e == prev_err:
                    logger.warning(e)
                    prev_err = e
            except KeyError:
                pass
        load_data.save_processed_cand_data()

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
        dataset = {"X": X.tolist(), "Y": np.array(Y).tolist(), "keys": np.array(list(data.keys())).tolist()}
        with open(processed_fpath, 'w') as file:
            json.dump(dataset, file, indent=4)
            if verbose:
                logger.success(f'File saved successfully to {processed_fpath}')

        logger.success(f'Loaded {len(X)} data samples into memory')
        return X, np.array(Y), list(data.keys())
    else:
        logger.info('Loading dataset from file')

        # Retrieve data from file
        # dataset = pd.read_csv(processed_fpath)
        with open(processed_fpath, 'r') as file:
            dataset = json.load(file)
        X = dataset["X"]
        Y = dataset["Y"]
        keys = dataset["keys"]

        logger.success(f'Data was loaded successfully from {processed_fpath}.')
        return X, Y, keys


