import logging
import joblib
import pandas as pd
from typing import Optional, Tuple

PATH = ""

DATA_VALID_PATH = "data/02_intermediate/data_valid.parquet.gzip"
RECOMMENDER_PATH = "models/recommender_v1"
USER_FEATURES_TRANSFORMED_PATH = (
    "data/03_primary/user_features_transformed.parquet.gzip"
)
ITEM_FEATURES_TRANSFORMED_PATH = (
    "data/03_primary/item_features_transformed.parquet.gzip"
)
USER_ITEM_FEATURES_PATH = "data/04_feature/user_item_features.parquet.gzip"


def load_inference_artifacts(
    path: Optional[str] = None,
    data_valid_path: Optional[str] = None,
    recommender_path: Optional[str] = None,
    user_features_transformed_path: Optional[str] = None,
    item_features_transformed_path: Optional[str] = None,
    user_item_features_path: Optional[str] = None,
) -> Tuple:
    """
    Loads recommender, user_features_transformed, item_features_transformed,
    user_item_features for inference
    """
    if path is None:
        path = PATH

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
        data_valid,
        recommender,
        user_features_transformed,
        item_features_transformed,
        user_item_features,
    )
