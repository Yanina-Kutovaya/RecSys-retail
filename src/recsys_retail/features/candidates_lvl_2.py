import logging
import numpy as np
import pandas as pd
from typing import Optional
from src.recsys_retail.features.recommenders import MainRecommender


logger = logging.getLogger(__name__)

__all__ = ['getting_candidates_for_level_2']

N_ITEMS = 100


def get_candidates(
    recommender,
    data_train_lvl_1: pd.DataFrame,
    data_val_lvl_1: pd.DataFrame, 
    data_val_lvl_2: Optional[pd.DataFrame] = None, 
    n_items: Optional[int] = None    
    ) -> pd.DataFrame:

    """
    Selects candidates for level 2 with n_items to be recommended
    for each of the candidate (e.g: top-100 items).        
    """
    
    users_train = data_train_lvl_1['user_id'].unique()
    users_valid = data_val_lvl_1['user_id'].unique().tolist()
    if data_val_lvl_2 is not None:
        users_test = data_val_lvl_2['user_id'].unique()
        add_to_valid = list(set(users_test) - (set(users_valid)))
        if add_to_valid:
            users_valid += add_to_valid

    current_users = list(set(users_valid) & set(users_train))    
    new_users = list(set(users_valid) - set(users_train))

    logging.info('Generating preference list for each user...')

    if n_items is None:
        n_items = N_ITEMS

    df = pd.DataFrame(users_valid, columns=['user_id'])
    cond_1 = df['user_id'].isin(current_users)
    df.loc[cond_1, 'candidates'] = df.loc[cond_1, 'user_id'].apply(
        lambda x: recommender.get_own_recommendations(x, n_items)
    )
    if new_users:
        cond_2 = df['user_id'].isin(new_users)
        df.loc[cond_2, 'candidates'] = df.loc[cond_2, 'user_id'].apply(
            lambda x: recommender.overall_top_purchases[:n_items]
        )
    users_lvl_2 = df

    return users_lvl_2
