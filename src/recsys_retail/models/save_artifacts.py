import logging
import joblib
import pandas as pd
from typing import NoReturn, Optional

__all__ = ['save_artifacts']

logger = logging.getLogger()

PATH = '../'

FOLDER_1 = 'data/01_raw/'
PREFILTERED_DATA_PATH = FOLDER_1 + 'data_prefiltered.parquet.gzip'

FOLDER_2 = 'data/02_intermediate/'
TRAIN_DATA_LEVEL_1_PATH = FOLDER_2 + 'data_train.parquet.gzip'
VALID_DATA_LEVEL_1_PATH = FOLDER_2 + 'data_valid.parquet.gzip'
VALID_DATA_LEVEL_2_PATH = FOLDER_2 + 'data_test.parquet.gzip'

FOLDER_3 = 'data/03_primary/'
ITEM_FEATURES_TRANSFORMED_PATH = FOLDER_3 + 'item_features_transformed.parquet.gzip'
USER_FEATURES_TRANSFORMED_PATH = FOLDER_3 + 'user_features_transformed.parquet.gzip'
DATA_TRAIN_LVL_1_PATH = FOLDER_3 + 'data_train_lvl_1_preprocessed.parquet.gzip'

FOLDER_4 = 'data/04_feature/'
CANDIDATES_PATH = FOLDER_4 + 'candidates_lvl_2.parquet.gzip'
USER_ITEM_FEATURES_PATH = FOLDER_4 + 'user_item_features.parquet.gzip'

FOLDER_5 = 'data/05_model_input/'
TRAIN_DATASET_LVL_2_PATH = FOLDER_5 + 'train_dataset_lvl_2.parquet.gzip'

FOLDER_6 = 'models/'
RECOMMENDER_PATH = FOLDER_6 + 'recommender_v1'


def save_prefiltered_data(
    data: pd.DataFrame,
    path: Optional[str] = None,
    prefilted_data_path: Optional[str] = None
    ) -> NoReturn:

    logging.info('Saving prefiltered data...') 

    if path is None:
        path = PATH

    if prefilted_data_path is None:
        prefilted_data_path = path + PREFILTERED_DATA_PATH
    data.to_parquet(prefilted_data_path, compression='gzip')


def save_time_split(
    data_train_lvl_1: pd.DataFrame, 
    data_val_lvl_1: pd.DataFrame, 
    data_val_lvl_2: Optional [pd.DataFrame] = None,
    path: Optional[str] = None,
    train_data_lvl_1_path: Optional[str] = None,
    valid_data_level_1_path: Optional[str] = None,
    valid_data_level_2_path: Optional[str] = None 
    ) -> NoReturn:

    logging.info('Saving time splitted data...')

    if path is None:
        path = PATH 
    
    if train_data_lvl_1_path is None:
        train_data_lvl_1_path = path + TRAIN_DATA_LEVEL_1_PATH
    data_train_lvl_1.to_parquet(
        train_data_lvl_1_path, compression='gzip'
    )
    if valid_data_level_1_path is None:
        valid_data_level_1_path = path + VALID_DATA_LEVEL_1_PATH
    data_val_lvl_1.to_parquet(
        valid_data_level_1_path, compression='gzip'
    )
    if data_val_lvl_2:
        if valid_data_level_2_path is None:
            valid_data_level_2_path = path + VALID_DATA_LEVEL_2_PATH
        data_val_lvl_2.to_parquet(
            valid_data_level_2_path, compression='gzip'
        )


def save_item_featutes(
    item_features_transformed: pd.DataFrame,
    path: Optional[str] = None,
    item_features_transformed_path: Optional[str] = None    
    ) -> NoReturn:

    logging.info('Saving item features...')

    if path is None:
        path = PATH

    if item_features_transformed_path is None:
        item_features_transformed_path = path + ITEM_FEATURES_TRANSFORMED_PATH
    item_features_transformed.to_parquet(
        item_features_transformed_path, compression='gzip'
    )


def save_user_features(    
    user_features_transformed: pd.DataFrame,
    path: Optional[str] = None,    
    user_features_transformed_path: Optional[str] = None
    ) -> NoReturn:

    logging.info('Saving user features...')

    if path is None:
        path = PATH

    if user_features_transformed_path is None:
        user_features_transformed_path = path + USER_FEATURES_TRANSFORMED_PATH
    user_features_transformed.to_parquet(
        user_features_transformed_path, compression='gzip'
    )


def save_preprocessed_lvl_1_train_dataset(
    data_train_lvl_1: pd.DataFrame,
    path: Optional[str] = None,
    data_train_lvl_1_path: Optional[str] = None    
    ) -> NoReturn:

    logging.info('Saving preprocessed level 1 train dataset...')

    if path is None:
        path = PATH

    if data_train_lvl_1_path is None:
        data_train_lvl_1_path = path + DATA_TRAIN_LVL_1_PATH
    data_train_lvl_1.to_parquet(
        data_train_lvl_1_path, compression='gzip'
    )


def save_recommender(
    recommender,
    path: Optional[str] = None,
    recommender_path: Optional[str] = None        
    ) -> NoReturn: 

    logging.info('Saving recommender...')

    if path is None:
        path = PATH 

    if recommender_path is None:
        recommender_path = path + RECOMMENDER_PATH
    joblib.dump(recommender, recommender_path, 3)    


def save_candidates(
    users_lvl_2: pd.DataFrame,
    path: Optional[str] = None,
    candidates_path: Optional[str] = None
    ) -> NoReturn:

    logging.info('Saving candidates for level 2 model...')

    if path is None:
        path = PATH

    if candidates_path is None:
        candidates_path = path + CANDIDATES_PATH
    users_lvl_2.to_parquet(candidates_path, compression='gzip')


def save_user_item_features(
    user_item_features: pd.DataFrame,
    path: Optional[str] = None,
    user_item_features_path: Optional[str] = None
    ) -> NoReturn:
    
    logging.info('Saving new user-item features...')

    if path is None:
        path = PATH
    
    if user_item_features_path is None:
        user_item_features_path = path + USER_ITEM_FEATURES_PATH
    user_item_features.to_parquet(
        user_item_features_path, compression='gzip'
    )


def save_train_dataset_lvl_2(
    train_dataset_lvl_2: pd.DataFrame,
    path: Optional[str] = None,
    train_dataset_lvl_2_path: Optional[str] = None
    ) -> NoReturn:

    logging.info('Saving train dataset for level 2 model...')

    if path is None:
        path = PATH

    if train_dataset_lvl_2_path is None:
        train_dataset_lvl_2_path = path + TRAIN_DATASET_LVL_2_PATH
    train_dataset_lvl_2.to_parquet(
        train_dataset_lvl_2_path, compression='gzip'
    ) 