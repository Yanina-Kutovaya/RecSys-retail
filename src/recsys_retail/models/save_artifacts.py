import logging
import joblib
import pandas as pd
from typing import NoReturn, Optional

__all__ = ['save_artifacts']

logger = logging.getLogger()

YC_BACKET = 's3a://recsys-retail/'

FOLDER_1 = 'data/01_raw/'
PREFILTERED_DATA_PATH = YC_BACKET + FOLDER_1 + 'data_prefiltered.csv.zip'

FOLDER_2 = 'data/02_intermediate/'
TRAIN_DATA_LEVEL_1_PATH = YC_BACKET + FOLDER_2 + 'data_train.csv.zip'
VALID_DATA_LEVEL_1_PATH = YC_BACKET + FOLDER_2 + 'data_valid.csv.zip'
VALID_DATA_LEVEL_2_PATH = YC_BACKET + FOLDER_2 + 'data_test.csv.zip'


FOLDER_3 = 'data/03_primary/'
ITEM_FEATURES_TRANSFORMED_PATH = YC_BACKET + FOLDER_3 + 'item_features_transformed.csv.zip'
USER_FEATURES_TRANSFORMED_PATH = YC_BACKET + FOLDER_3 + 'user_features_transformed.csv.zip'
DATA_TRAIN_LVL_1_PATH = YC_BACKET + FOLDER_3 + 'data_train_lvl_1_preprocessed.csv.zip'

FOLDER_4 = 'data/04_feature/'
CANDIDATES_PATH = YC_BACKET + FOLDER_4 + 'candidates_lvl_2.csv.zip'
USER_ITEM_FEATURES_PATH = YC_BACKET + FOLDER_4 + 'user_item_features.csv.zip'

FOLDER_5 = 'data/05_model_input/'
TRAIN_DATASET_LVL_2_PATH = YC_BACKET + FOLDER_5 + 'train_dataset_lvl_2.csv.zip'

FOLDER_6 = 'models/'
RECOMMENDER_PATH = YC_BACKET + FOLDER_6 + 'recommender_v1'


def save_prefiltered_data(
    data: pd.DataFrame,
    prefilted_data_path: Optional[str] = None
    ) -> NoReturn:

    logging.info('Saving prefiltered data...') 

    if prefilted_data_path is None:
        prefilted_data_path = PREFILTERED_DATA_PATH
    data.to_csv(prefilted_data_path, index=False, compression='zip')


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