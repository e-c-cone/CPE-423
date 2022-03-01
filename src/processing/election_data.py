import os
import pandas as pd


def get_winner_id(candidates: pd.DataFrame, year: int = 1995, state_fips: int = 1) -> int:
    """
    Identifies the winning political candidate from a specified year and state and returns the candidate_id
    :param candidates:
    :param year:
    :param state_fips:
    :return:
    """
    candidates = candidates[candidates['year'] == year]
    candidates = candidates[candidates['state_fips'] == state_fips]
    # print(candidates)
    candidates = candidates[candidates['candidatevotes'] == candidates['candidatevotes'].max()]
    print(candidates['candidate'])
    return candidates['candidate']
