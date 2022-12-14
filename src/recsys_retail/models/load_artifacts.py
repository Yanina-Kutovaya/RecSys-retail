import logging
import joblib
import pandas as pd
from typing import Optional, Tuple

PATH = ""

FOLDER = "data/04_feature/"
CURRENT_USER_LIST_PATH = FOLDER + "current_user_list.joblib"
DATA_VALID_PATH = FOLDER + "data_valid.parquet.gzip"
RECOMMENDER_PATH = FOLDER + "recommender_v1.joblib"
ITEM_FEATURES_TRANSFORMED_PATH = FOLDER + "item_features_transformed.parquet.gzip"
USER_FEATURES_TRANSFORMED_PATH = FOLDER + "user_features_transformed.parquet.gzip"
USER_ITEM_FEATURES_PATH = FOLDER + "user_item_features.parquet.gzip"


def load_inference_artifacts(
    path: Optional[str] = None,
    current_user_list_path: Optional[str] = None,
    data_valid_path: Optional[str] = None,
    recommender_path: Optional[str] = None,
    user_features_transformed_path: Optional[str] = None,
    item_features_transformed_path: Optional[str] = None,
    user_item_features_path: Optional[str] = None,
) -> Tuple:
    """
    Loads current_user_list, data_valid, recommender, user_features_transformed,
    item_features_transformed, user_item_features for inference
    """
    if path is None:
        path = PATH

    if current_user_list_path is None:
        current_user_list_path = CURRENT_USER_LIST_PATH
    current_user_list = joblib.load(current_user_list_path)

    if data_valid_path is None:
        data_valid_path = path + DATA_VALID_PATH
    data_valid = pd.read_parquet(data_valid_path)

    if recommender_path is None:
        recommender_path = path + RECOMMENDER_PATH
    recommender = joblib.load(recommender_path)

    if user_features_transformed_path is None:
        user_features_transformed_path = path + USER_FEATURES_TRANSFORMED_PATH
    user_features_transformed = pd.read_parquet(user_features_transformed_path)

    if item_features_transformed_path is None:
        item_features_transformed_path = path + ITEM_FEATURES_TRANSFORMED_PATH
    item_features_transformed = pd.read_parquet(item_features_transformed_path)

    if user_item_features_path is None:
        user_item_features_path = path + USER_ITEM_FEATURES_PATH
    user_item_features = pd.read_parquet(user_item_features_path)

    return (
        current_user_list,
        data_valid,
        recommender,
        user_features_transformed,
        item_features_transformed,
        user_item_features,
    )
