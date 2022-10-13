import logging
import numpy as np
import pandas as pd
from typing import Optional
from src.recsys_retail.features.recommenders import MainRecommender


logger = logging.getLogger(__name__)

__all__ = ['getting_candidates_for_level_2']


N_CANDIDATES = 300
PATH = 'data/04_feature/'
CANDIDATES_PATH = PATH + 'candidates_lvl_2.csv.zip'

def get_candidates(
    data_train_lvl_1: pd.DataFrame,
    data_val_lvl_1: pd.DataFrame, 
    data_val_lvl_2: pd.DataFrame, 
    n_candidates: Optional[int] = None,
    candidates_path: Optional[str] = None
    ):

    """
    Selects candidates for level 2 and long list of preferred items for each user.
    Later the model will select short lists of items (top-5) from the long lists.        
    """
    
    recommender = MainRecommender(data_train_lvl_1)

    users_lvl_1 = data_train_lvl_1['user_id'].unique()
    users_lvl_2 = data_train_lvl_2['user_id'].unique().tolist()
    users_lvl_3 = data_val_lvl_2['user_id'].unique()

    add_to_lvl_2 = list(set(users_lvl_3) - (set(users_lvl_2)))
    if add_to_lvl_2:
        users_lvl_2 += add_to_lvl_2

    current_users = list(set(users_lvl_2) & set(users_lvl_1))    
    new_users = list(set(users_lvl_2) - set(users_lvl_1))

    if n_candidates is None:
        n_candidates = N_CANDIDATES

    df = pd.DataFrame(users_lvl_2, columns=['user_id'])
    cond_1 = df['user_id'].isin(current_users)
    df.loc[cond_1, 'candidates'] = df.loc[cond_1, 'user_id'].apply(
        lambda x: recommender.get_own_recommendations(x, n_candidates)
    )
    if new_users:
        cond_2 = df['user_id'].isin(new_users)
        df.loc[cond_2, 'candidates'] = df.loc[cond_2, 'user_id'].apply(
                lambda x: recommender.overall_top_purchases[:n_candidates]
        )
    if candidates_path is None:
        candidates_path = CANDIDATES_PATH
    df.to_csv(candidates_path, compression='zip')

    return df, recommender