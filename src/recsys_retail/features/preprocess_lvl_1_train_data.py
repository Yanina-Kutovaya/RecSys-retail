import os
import logging
import numpy as np
import pandas as pd
from pickle import dump, load
from typing import Optional

from src.recsys_retail.data.make_dataset import load_data
from src.recsys_retail.features.prefilter import prefilter_items
from src.recsys_retail.features.user_features import transform_user_features
from src.recsys_retail.features.item_features import (
    fit_transform_item_features, transform_item_features
)

logger = logging.getLogger(__name__)

__all__ = ['preprocess_lvl_1_data']

PATH = 'data/03_primary/'
TRAIN_DATA_LEVEL_1_PATH = PATH + 'data_train_lvl_1.csv.zip'

def get_lvl_1_train_dataset(
    data_train_lvl_1: pd.DataFrame,
    item_features: pd.DataFrame,
    user_features:pd.DataFrame,
    train_data_lvl_1_path: Optional[str] = None
    ) -> pd.DataFrame:

    """ 
    Prepares dataset which combines transactions, user features 
    and item features to be passed to user-item matrix for recommender.
    """

    logging.info('Preprocessing level 1 train dataset...')

    # Prefilter items
    data_train_lvl_1 = prefilter_items(data_train_lvl_1, item_features)

    # Preprocess user features and merge with transactions data    
    user_features_transformed = transform_user_features(user_features)
    data_train_lvl_1 = pd.merge(
        data_train_lvl_1, user_features_transformed, on='user_id', how='left'
    )
    # Preprocess item features and merge with transactions data
    item_features_transformed = fit_transform_item_features(item_features)
    data_train_lvl_1 = pd.merge(
        data_train_lvl_1, item_features_transformed, on='item_id', how='left'
    )
    # Save results    
    if train_data_lvl_1_path is None:
        train_data_lvl_1_path = TRAIN_DATA_LEVEL_1_PATH
    data_train_lvl_1.to_csv(train_data_lvl_1_path, compression='zip')
   
    return data_train_lvl_1