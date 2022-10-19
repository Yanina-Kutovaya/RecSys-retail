import pandas as pd
import joblib
from typing import Optional

N_ITEMS = 100

PATH_1 = 'models/'
RECOMMENDER_PATH = PATH_1 + 'recommender_v1' 

PATH_2 = 'data/01_raw/'
DATA_TRAIN_PATH = PATH_2 + 'data_train.csv.zip'
DATA_VALID_PATH = PATH_2 + 'data_valid.csv.zip'

PATH_3 = 'data/02_intermediate/'
USER_FEATURES_TRANSFORMED_PATH = PATH_3 + 'user_features_transformed.csv.zip'
ITEM_FEATURES_TRANSFORMED_PATH = PATH_3 + 'item_features_transformed.csv.zip'

PATH_4 = 'data/04_feature/' 
USER_ITEM_FEATURES_FINAL_PATH = PATH_4 + 'user_item_features_final.csv.zip'


def preprocess(
    test: pd.DataFrame,
    
    n_items: Optional[int] = None
    ) -> pd.DataFrame:

    """
    Preprocesses transactions data for inference with binary classifier
    """ 

    (
        recommender, 
        data_train, data_valid, 
        item_features_transformed, 
        user_features_transformed,    
        user_item_features_final
    ) = load_artefacts()

    if n_items is None:
        n_items = N_ITEMS

    users_inference = get_candidates(
        recommender, data_train, test, n_items
    )
    test_dataset_inference = get_targets_lvl_2(
        users_inference, 
        data_valid,
        item_features_transformed, 
        user_features_transformed,    
        user_item_features_final,     
        n_items
    )

    return test_dataset_inference.drop('target', axis=1).fillna(0)


def load_artefacts(
    recommender_path: Optional[str] = None,
    data_train_path: Optional[str] = None,
    data_valid_path: Optional[str] = None,
    item_features_transformed_path: Optional[str] = None,
    user_features_transformed_path: Optional[str] = None,
    user_item_features_final_path: Optional[str] = None,
    ):

    """
    Loads artefacts for preprocess function
    """

    if recommender_path is None:
        recommender_path = RECOMMENDER_PATH
    recommender = joblib.load(recommender_path)

    if data_train_path is None:
        data_train_path = DATA_TRAIN_PATH
    data_train = pd.read_csv(data_train_path)

    if data_valid_path is None:
        data_valid_path = DATA_VALID_PATH
    data_valid = pd.read_csv(data_valid_path)

    if item_features_transformed_path is None:
        item_features_transformed_path = ITEM_FEATURES_TRANSFORMED_PATH
    item_features_transformed = pd.read_csv(item_features_transformed_path)

    if user_features_transformed_path is None:
        user_features_transformed_path = USER_FEATURES_TRANSFORMED_PATH
    user_features_transformed = pd.read_csv(user_features_transformed_path)

    if user_item_features_final_path is None:
        user_item_features_final_path = USER_ITEM_FEATURES_FINAL_PATH
    user_item_features_final = pd.read_csv(user_item_features_final_path)

    return (
        recommender, 
        data_train, data_valid, 
        item_features_transformed, 
        user_features_transformed,    
        user_item_features_final
    ) 