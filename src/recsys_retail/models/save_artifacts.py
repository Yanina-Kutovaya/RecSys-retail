import logging
import pandas as pd
from typing import NoReturn, Optional

__all__ = ['save_artifacts']

logger = logging.getLogger()

PATH = 'data/01_raw/'
TRAIN_DATA_LEVEL_1_PATH = PATH + 'data_train.csv.zip'
VALID_DATA_LEVEL_1_PATH = PATH + 'data_valid.csv.zip'
VALID_DATA_LEVEL_2_PATH = PATH + 'data_test.csv.zip'

PATH = 'data/02_intermediate/'
PREFILTERED_TRAIN_DATA_LEVEL_1_PATH = PATH + 'data_train_lvl_1_prefiltered.csv.zip'
ITEM_FEATURES_TRANSFORMED_PATH = PATH + 'item_features_transformed.csv.zip'
USER_FEATURES_TRANSFORMED_PATH = PATH + 'user_features_transformed.csv.zip'

PATH = 'data/03_primary/'
DATA_TRAIN_LVL_1_PATH = PATH + 'data_train_lvl_1_preprocessed.csv.zip'

PATH = 'data/04_feature/'
CANDIDATES_PATH = PATH + 'candidates_lvl_2.csv.zip'
USER_ITEM_FEATURES_PATH = PATH + 'user_item_features.csv.zip'

PATH = 'data/05_model_input/'
TRAIN_DATASET_LVL_2_PATH = PATH + 'train_dataset_lvl_2.csv.zip'

PATH = 'models/'
RECOMMENDER_PATH = PATH + 'recommender_v1'


def save_time_split(
    data_train_lvl_1: pd.DataFrame, 
    data_val_lvl_1: pd.DataFrame, 
    data_val_lvl_2: Optional [pd.DataFrame] = None,
    train_data_lvl_1_path: Optional[str] = None,
    valid_data_level_1_path: Optional[str] = None,
    valid_data_level_2_path: Optional[str] = None 
    ) -> NoReturn:

    logging.info('Saving time splitted data...') 
    
    if train_data_lvl_1_path is None:
        train_data_lvl_1_path = TRAIN_DATA_LEVEL_1_PATH
    data_train_lvl_1.to_csv(
        train_data_lvl_1_path, index=False, compression='zip'
    )
    if valid_data_level_1_path is None:
        valid_data_level_1_path = VALID_DATA_LEVEL_1_PATH
    data_val_lvl_1.to_csv(
        valid_data_level_1_path, index=False, compression='zip'
    )
    if data_val_lvl_2:
        if valid_data_level_2_path is None:
            valid_data_level_2_path = VALID_DATA_LEVEL_2_PATH
        data_val_lvl_2.to_csv(
            valid_data_level_2_path, index=False, compression='zip'
        )


def save_prefiltered_data(
    data: pd.DataFrame,
    prefilted_train_data_lvl_1_path: Optional[str] = None
    ) -> NoReturn:

    logging.info('Saving prefiltered data...') 

    if prefilted_train_data_lvl_1_path is None:
        prefilted_train_data_lvl_1_path = PREFILTERED_TRAIN_DATA_LEVEL_1_PATH
    data.to_csv(prefilted_train_data_lvl_1_path, index=False, compression='zip')


def save_item_featutes(
    item_features_transformed: pd.DataFrame,
    item_features_transformed_path: Optional[str] = None    
    ) -> NoReturn:

    logging.info('Saving item features...')

    if item_features_transformed_path is None:
        item_features_transformed_path = ITEM_FEATURES_TRANSFORMED_PATH
    item_features_transformed.to_csv(
        item_features_transformed_path, index=False, compression='zip'
    )


def save_user_features(    
    user_features_transformed: pd.DataFrame,    
    user_features_transformed_path: Optional[str] = None
    ) -> NoReturn:

    logging.info('Saving user features...')

    if user_features_transformed_path is None:
        user_features_transformed_path = USER_FEATURES_TRANSFORMED_PATH
    user_features_transformed.to_csv(
        user_features_transformed_path, index=False, compression='zip'
    )


def save_preprocessed_lvl_1_train_dataset(
    data_train_lvl_1: pd.DataFrame,
    data_train_lvl_1_path: Optional[str] = None    
    ) -> NoReturn:

    logging.info('Saving preprocessed level 1 train dataset...')

    if data_train_lvl_1_path is None:
        data_train_lvl_1_path = DATA_TRAIN_LVL_1_PATH
    data_train_lvl_1.to_csv(
        data_train_lvl_1_path, index=False, compression='zip'
    )


def save_recommender(
    recommender,
    recommender_path: Optional[str] = None        
    ) -> NoReturn: 

    logging.info('Saving recommender...') 

    if recommender_path is None:
        recommender_path = RECOMMENDER_PATH
    joblib.dump(recommender, recommender_path, 3)    


def save_candidates(
    users_lvl_2: pd.DataFrame,
    candidates_path: Optional[str] = None
    ) -> NoReturn:

    logging.info('Saving candidates for level 2 model...')

    if candidates_path is None:
        candidates_path = CANDIDATES_PATH
    users_lvl_2.to_csv(candidates_path, index=False, compression='zip')


def save_user_item_features(
    user_item_features: pd.DataFrame,
    user_item_features_path: Optional[str] = None
    ) -> NoReturn:
    
    logging.info('Saving new user-item features...')
    
    if user_item_features_path is None:
        user_item_features_path = USER_ITEM_FEATURES_PATH
    user_item_features.to_csv(
        user_item_features_path, index=False, compression='zip'
    )


def save_train_dataset_lvl_2(
    train_dataset_lvl_2: pd.DataFrame,
    train_dataset_lvl_2_path: Optional[str] = None
    ) -> NoReturn:

    logging.info('Saving train dataset for level 2 model...')

    if train_dataset_lvl_2_path is None:
        train_dataset_lvl_2_path = TRAIN_DATASET_LVL_2_PATH
    train_dataset_lvl_2.to_csv(
        train_dataset_lvl_2_path, index=False, compression='zip'
    ) 