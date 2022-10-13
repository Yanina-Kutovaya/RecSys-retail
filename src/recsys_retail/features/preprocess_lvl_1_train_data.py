import os
import logging
import numpy as np
import pandas as pd
from pickle import dump, load

from src.recsys_retail.data.make_dataset import load_data
from src.recsys_retail.features.prefilter import prefilter_items
from src.recsys_retail.features.user_features import transform_user_features
from src.recsys_retail.features.item_features import (
    fit_transform_item_features, transform_item_features
)

logger = logging.getLogger(__name__)

__all__ = ['preprocess_lvl_1_data']

TRAIN_DATA_LEVEL_1_PATH = 'data_train_lvl_1.csv'


def get_lvl_1_train_dataset(train_data_lvl_1_path = None) -> pd.DataFrame:

    logging.info('Preprocessing level 1 train dataset...')

    data, item_features, user_features = load_data()

    # For the 1st level we use older data leaving 9 weeks for validation:
    # 6 weeks for the 1st level validation and 3 weeks for the 2nd level.
    val_lvl_1_size_weeks = 6
    val_lvl_2_size_weeks = 3
    t0 = data['week_no'].max() - (val_lvl_1_size_weeks + val_lvl_2_size_weeks)
    data_train_lvl_1 = data[data['week_no'] < t0]

    # Prefilter items
    data_train_lvl_1 = prefilter_items(data_train_lvl_1, item_features)

    # Preprocess user features and merge with transactions data
    os.chdir('../data/02_intermediate')
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
    os.chdir('../03_primary')
    if train_data_lvl_1_path is None:
        train_data_lvl_1_path = TRAIN_DATA_LEVEL_1_PATH
    data_train_lvl_1.to_csv(train_data_lvl_1_path)

    return data_train_lvl_1