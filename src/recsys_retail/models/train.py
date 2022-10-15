import logging
import pandas as pd
from typing import Optional, Tuple

from src.recsys_retail.data.make_dataset import load_data
from src.recsys_retail.features.data_time_split import time_split
from src.recsys_retail.features.preprocess_lvl_1_train_data import get_lvl_1_train_dataset
from src.recsys_retail.features.candidates_lvl_2 import get_candidates
from src.recsys_retail.features.targets import get_targets_lvl_2


logger = logging.getLogger(__name__)

__all__ = ['generate_lvl_2_dataset']

N_ITEMS = 300
PATH_1 = 'data/02_intermediate/'
ITEM_FEATURES_TRANSFORMED_PATH = PATH_1 + 'item_features_transformed.csv.zip'
USER_FEATURES_TRANSFORMED_PATH = PATH_1 + 'user_features_transformed.csv.zip'

PATH_2 = 'data/05_model_input'
TRAIN_DATASET_LVL_2_PATH = PATH_2 + 'dataset_lvl_2.csv.zip'


def get_train_dataset_lvl_2(
    data: pd.DataFrame, 
    item_features: pd.DataFrame, 
    user_features: pd.DataFrame,
    n_items: Optional[int] = None,
    item_features_transformed_path: Optional[str] = None,
    user_features_transformed_path: Optional[str] = None,
    user_item_features_path: Optional[str] = None,
    train_dataset_lvl_2_path: Optional[str] = None
    ) -> pd.DataFrame:

    logging.info('Generating level 1 dataset...')

    data_train_lvl_1, data_train_lvl_2, data_val_lvl_2 = time_split(data)
    data_train_lvl_1 = get_lvl_1_train_dataset(data_train_lvl_1, item_features, user_features)

    logging.info('Generating level 2 dataset...')

    if n_items is None:
        n_items = N_ITEMS
    users_lvl_2, recommender = get_candidates(
        data_train_lvl_1, data_train_lvl_2, data_val_lvl_2, n_items
    )
    logging.info('Generating new user-item features...')
    user_item_features = get_user_item_features(recommender, data_train_lvl_1)

    item_features_transformed, user_features_transformed = load_transformed_features()   
    
    train_dataset_lvl_2 = get_targets_lvl_2(
        users_lvl_2, data_train_lvl_2, item_features_transformed, 
        user_features_transformed, user_item_features
    )

    logging.info('Saving level 2 dataset...')

    if train_dataset_lvl_2_path is None:
        train_dataset_lvl_2_path = TRAIN_DATASET_LVL_2_PATH
    train_dataset_lvl_2.to_csv(
        train_dataset_lvl_2_path, index=False, compression='zip'
    )

    return dataset_lvl_2

def load_transformed_features(
    item_features_transformed_path: Optional[str] = None,
    user_features_transformed_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logging.info(
        f'Reading item_features_transformed_path from {item_features_transformed_path}...'
    )
    if item_features_transformed_path is None:
        item_features_transformed_path = ITEM_FEATURES_TRANSFORMED_PATH    
    item_features_transformed = pd.read_csv(
        item_features_transformed_path, compression='zip'
    )
    logging.info(
        f'Reading user_features_transformed_path from {user_features_transformed_path}...'
    )
    if user_features_transformed_path is None:
        user_features_transformed_path = USER_FEATURES_TRANSFORMED_PATH    
    user_features_transformed = pd.read_csv(
        user_features_transformed_path, compression='zip'
    )
    return item_features_transformed, user_features_transformed