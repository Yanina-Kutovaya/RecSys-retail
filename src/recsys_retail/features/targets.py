import logging
import numpy as np
import pandas as pd
from pickle import dump
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ['generate_targets']


CANDIDATES_PATH = 'data/04_feature/candidates_lvl_2.csv.zip'
VALID_DATA_LEVEL_1_PATH = 'data/01_raw/data_val_lvl_1.csv.zip'
ITEM_FEATURES_TRANSFORMED_PATH = 'data/02_intermediate/item_features_transformed.csv.zip'
USER_FEATURES_TRANSFORMED_PATH = 'data/02_intermediate/user_features_transformed.csv.zip'
USER_ITEM_FEATURES_PATH = 'data/04_feature/user_item_features.csv.zip'
N_RECOMENDATIONS = 5

PATH = 'data/05_model_input/'
TARGET_LVL_2_PATH = PATH + 'targets_lvl_2.csv.zip'


def get_targets_lvl_2(
    users_lvl_2_path: Optional[str] = None, 
    data_val_lvl_1_path: Optional[str] = None,
    item_features_transformed_path: Optional[str] = None,
    user_features_transformed_path: Optional[str] = None,
    user_item_features_path: Optional[str] = None,
    n_recommendations: Optional[int] = None,
    targets_lvl_2_path: Optional[str] = None 
    ) -> pd.DataFrame:
    """ 
    Generates targets for candidates from level 2 (data_train_lvl_2 6-week period,
    which was validation period for level 1).
    """
    
    logging.info('Generating targets for level 2...')

    if users_lvl_2_path is None:
        users_lvl_2_path = CANDIDATES_PATH
    logging.info(f'Reading users_lvl_2_path from {users_lvl_2_path}...')    
    users_lvl_2 = pd.read_csv(users_lvl_2_path, compression='zip')

    if n_recommendations is None:
        n_recommendations = N_RECOMENDATIONS

    df = pd.DataFrame(
        {'user_id': users_lvl_2['user_id'].values.repeat(n_recommendations),
        'item_id': np.concatenate(users_lvl_2['candidates'].values)
        }
    )
    if data_val_lvl_1_path is None:
        data_val_lvl_1_path = VALID_DATA_LEVEL_1_PATH
    logging.info(f'Reading data_val_lvl_1_path from {data_val_lvl_1_path}...')
    data_train_lvl_2 = pd.read_csv(data_val_lvl_1_path, compression='zip')
    
    targets_lvl_2 = data_train_lvl_2[['user_id', 'item_id']].copy()
    targets_lvl_2['target'] = 1  # Here we have only purchases
    targets_lvl_2 = df.merge(targets_lvl_2, on=['user_id', 'item_id'], how='left')
    targets_lvl_2['target'].fillna(0, inplace= True)

    if item_features_transformed_path is None:
        item_features_transformed_path = ITEM_FEATURES_TRANSFORMED_PATH
    logging.info(
        f'Reading item_features_transformed_path from {item_features_transformed_path}...'
    )
    item_features_transformed = pd.read_csv(item_features_transformed_path, compression='zip')
    targets_lvl_2 = targets_lvl_2.merge(
        item_features_transformed, on='item_id', how='left'
    )
    if user_features_transformed_path is None:
        user_features_transformed_path = USER_FEATURES_TRANSFORMED_PATH
    logging.info(
        f'Reading user_features_transformed_path from {user_features_transformed_path}...'
    )
    user_features_transformed = pd.read_csv(user_features_transformed_path, compression='zip')
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