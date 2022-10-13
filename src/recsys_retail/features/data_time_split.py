import os
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ['splitting_data_2_levels']

PATH = 'data/01_raw/'
TRAIN_DATA_LEVEL_1_PATH = PATH + 'data_train_lvl_1.csv.zip'
VALID_DATA_LEVEL_1_PATH = PATH + 'data_val_lvl_1.csv.zip'
VALID_DATA_LEVEL_2_PATH = PATH + 'data_val_lvl_2.csv.zip'


def time_split(
    data: pd.DataFrame,
    save_split = True,
    train_data_lvl_1_path: Optional[str] = None,
    valid_data_level_1_path: Optional[str] = None,
    valid_data_level_2_path: Optional[str] = None    
    ) -> pd.DataFrame:
    """ 
    Train-validation-test time split for two-stage recommender system.

    Train - validation - test schema:
    -- old purchases -- | -- 6 weeks-- | -- 3 weeks--
    
    For the 1st level we use older data leaving 9 weeks for validation:
    6 weeks for the 1st level validation and 3 weeks for the 2nd level.
    """

    logging.info('Splitting data for 2 levels of train-validation...')

    
    val_lvl_1_size_weeks = 6
    val_lvl_2_size_weeks = 3

    t0 = data['week_no'].max() - (val_lvl_1_size_weeks + val_lvl_2_size_weeks)
    t1 = data['week_no'].max() - val_lvl_2_size_weeks

    data_train_lvl_1 = data[data['week_no'] < t0]
    data_val_lvl_1 = data[(data['week_no'] >= t0) & (data['week_no'] < t1)]
    data_val_lvl_2 = data[data['week_no'] >= t1]

    if save_split:        
        if train_data_lvl_1_path is None:
            train_data_lvl_1_path = TRAIN_DATA_LEVEL_1_PATH
        data_train_lvl_1.to_csv(train_data_lvl_1_path, compression='zip')

        if valid_data_level_1_path is None:
            valid_data_level_1_path = VALID_DATA_LEVEL_1_PATH
        data_val_lvl_1.to_csv(valid_data_level_1_path, compression='zip')

        if valid_data_level_2_path is None:
            valid_data_level_2_path = VALID_DATA_LEVEL_2_PATH
        data_val_lvl_2.to_csv(valid_data_level_2_path, compression='zip')

    return data_train_lvl_1, data_val_lvl_1, data_val_lvl_2