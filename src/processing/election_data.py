import os
import time
import pandas as pd
import numpy as np
from loguru import logger
import utils
import processing.load_data as load_data


def generate_dataset(cutoff_year: int = 1990, verbose: bool = False):
    """

    :return:
    """
    logger.info('Generating viable dataset')

    FIPS = pd.read_csv(os.path.join('data', 'FIPS.csv'))
    FIPS_POSSIBLE = FIPS['fips'].to_numpy()
    house_rep = load_data.get_candidates(cutoff_year=cutoff_year)

    reps = []
    data = {}
    for state in FIPS_POSSIBLE:
        for year in range(cutoff_year, 2000)[::2]:  # TODO
            winner_ids = load_data.get_winner_ids(candidates=house_rep, year=year, state_fips=state, verbose=verbose)
            data[f'{state}_{year}'] = winner_ids
            if winner_ids:
                # logger.info(winner_ids)
                reps += winner_ids
    logger.success(f'There are {len(set(reps))} valid loaded representatives')
    # logger.info(data)

    logger.info('Merging datasets')
    X = []
    Y = []
    data_size = len(data.keys())
    start = time.time()
    for i, election in enumerate(data.keys()):
        if verbose and (i % int(data_size/20) == 0 or i == 10) and i > 0:
            dur = time.time() - start
            remaining = int(dur * (data_size - i) / i)
            remaining = time.strftime("%H:%M:%S", time.gmtime(remaining))
            logger.debug(f'Merging datasets... election {i} out of {data_size} with {len(data[election])} candidates:'
                         f'\testimated time {remaining=}')
        for j, candidate_id in enumerate(data[election]):
            if j <= 4:  # TODO
                state_fips = int(election.split('_')[0])
                year = int(election.split('_')[1])
                pop = load_data.get_population_data(year=year, state_fips=state_fips, verbose=verbose)
                inc = load_data.get_personal_income(year=year, state_fips=state_fips, verbose=verbose)
                rating = load_data.get_ratings(candidate_id, election.split('_')[1])
                # print(pop, inc)
                if pop and inc and rating:
                    X += [pop + inc]
                    Y += [rating]
    logger.success(f'Loaded {len(X)} data samples into memory')
    print(np.array(X).shape, np.array(Y).shape)
    return X, Y
