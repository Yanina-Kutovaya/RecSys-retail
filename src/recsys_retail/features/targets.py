import logging
import numpy as np
import pandas as pd
from pickle import dump
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ['generate_targets']


N_RECOMENDATIONS = 5

PATH = 'data/05_model_input/'
TARGET_LVL_2_PATH = PATH + 'targets_lvl_2.csv.zip'


def get_targets_lvl_2(
    users_lvl_2: pd.DataFrame, 
    data_train_lvl_2: pd.DataFrame,
    item_features_transformed: pd.DataFrame,  
    user_features_transformed: pd.DataFrame,
    user_item_features:pd.DataFrame,     
    n_recommendations: Optional[int] = None,
    targets_lvl_2_path: Optional[str] = None 
    ) -> pd.DataFrame:
    """ 
    Generates targets for candidates from level 2 (data_train_lvl_2 6-week period,
    which was validation period for level 1).
    """
    
    logging.info('Generating targets for level 2...')

    if n_recommendations is None:
        n_recommendations = N_RECOMENDATIONS

    df = pd.DataFrame(
        {'user_id': users_lvl_2['user_id'].values.repeat(n_recommendations),
        'item_id': np.concatenate(users_lvl_2['candidates'].values)
        }
    )       
    targets_lvl_2 = data_train_lvl_2[['user_id', 'item_id']].copy()
    targets_lvl_2['target'] = 1  # Here we have only purchases
    targets_lvl_2 = df.merge(targets_lvl_2, on=['user_id', 'item_id'], how='left')
    targets_lvl_2['target'].fillna(0, inplace= True)    
    
    targets_lvl_2 = targets_lvl_2.merge(
        item_features_transformed, on='item_id', how='left'
    )    
    targets_lvl_2 = targets_lvl_2.merge(
        user_features_transformed, on='user_id', how='left'
    )
    targets_lvl_2 = targets_lvl_2.merge(
        user_item_features, on=['user_id', 'item_id'], how='left'
    ).drop_duplicates()

    if targets_lvl_2_path is None:
        targets_lvl_2_path = TARGET_LVL_2_PATH
    targets_lvl_2.to_csv(targets_lvl_2_path, index=False, compression='zip')

    return targets_lvl_2