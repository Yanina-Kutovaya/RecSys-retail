import logging
import pandas as pd
from pickle import dump
from typing import Optional

from src.recsys_retail.data.make_dataset import load_data
from src.recsys_retail.features.data_time_split import time_split
from src.recsys_retail.features.prefilter import prefilter_items
from src.recsys_retail.features.user_features import fit_transform_user_features
from src.recsys_retail.features.item_features import fit_transform_item_features
from src.recsys_retail.models.train_recommender import train_save_recommender
from src.recsys_retail.features.candidates_lvl_2 import get_candidates
from src.recsys_retail.features.new_item_user_features import get_user_item_features
from src.recsys_retail.features.targets import get_targets_lvl_2


logger = logging.getLogger(__name__)

__all__ = ['preprocess_data']

PATH_1 = 'data/02_intermediate/'
DATA_TRAIN_LVL_1_PATH = PATH_1 + 'data_train_lvl_1_preprocessed.csv.zip'

N_ITEMS = 100
PATH_2 = 'data/05_model_input/'
TRAIN_DATASET_LVL_2_PATH = PATH_2 + 'train_dataset_lvl_2.csv.zip'


def data_preprocessing_pipeline(
    data: pd.DataFrame, 
    item_features: pd.DataFrame, 
    user_features: pd.DataFrame,
    n_items: Optional[int] = None,
    recommender_path: Optional[str] = None,
    train_dataset_lvl_2_path: Optional[str] = None
    ) -> pd.DataFrame:

    """
    Prepares dataset from transactions data, item features and user features
    to be used in binary classification models.

    n_items - the number of items selected by the recommender for each user 
              on the 1st stage (long list), the bases for the further short list
              selection with the binary classification model. 

    """

    logging.info('Splitting dataset for level 1, level 2 preprocessing...')
    data_train_lvl_1, data_train_lvl_2, data_val_lvl_2 = time_split(data)

    logging.info('Prefiltering transactions data...')
    data = prefilter_items(data, item_features)

    logging.info('Preprocessing level 1 train dataset...')
        
    user_features_transformed = fit_transform_user_features(user_features)
    data_train_lvl_1 = pd.merge(
        data_train_lvl_1, user_features_transformed, on='user_id', how='left'
    )    
    item_features_transformed = fit_transform_item_features(item_features)
    data_train_lvl_1 = pd.merge(
        data_train_lvl_1, item_features_transformed, on='item_id', how='left'
    )

    logging.info('Saving preprocessed level 1 train dataset...')

    if train_data_lvl_1_path is None:
        train_data_lvl_1_path = DATA_TRAIN_LVL_1_PATH
    data_train_lvl_1.to_csv(train_data_lvl_1_path, index=False, compression='zip')

    logging.info('Selecting users for level 2 dataset...')

    recommender = train_save_recommender(data_train_lvl_1)

    if n_items is None:
        n_items = N_ITEMS
    users_lvl_2 = get_candidates(
        recommender, data_train_lvl_1, data_train_lvl_2, data_val_lvl_2, n_items
    )
    logging.info('Generating new user-item features...') 

    user_item_features = get_user_item_features(recommender, data_train_lvl_1)       
    
    logging.info('Generating train dataset for level 2 model...')

    train_dataset_lvl_2 = get_targets_lvl_2(
        users_lvl_2, 
        data_train_lvl_2, 
        item_features_transformed, 
        user_features_transformed, 
        user_item_features, 
        n_items
    )
    logging.info('Saving train dataset for level 2 model...')

    if train_dataset_lvl_2_path is None:
        train_dataset_lvl_2_path = TRAIN_DATASET_LVL_2_PATH
    train_dataset_lvl_2.to_csv(
        train_dataset_lvl_2_path, index=False, compression='zip'
    )

    return train_dataset_lvl_2