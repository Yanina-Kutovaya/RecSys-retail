import logging
import joblib
import pandas as pd
from typing import Optional, Tuple

PATH = '../'
RECOMMENDER_PATH = 'models/recommender_v1'
USER_FEATURES_TRANSFORMED_PATH = 'data/03_primary/user_features_transformed.csv.zip'
ITEM_FEATURES_TRANSFORMED_PATH = 'data/03_primary/item_features_transformed.csv.zip'
USER_ITEM_FEATURES_PATH = 'data/04_feature/user_item_features.csv.zip'

def load_inference_artifacts(
    path: Optional[str] = None,    
    recommender_path: Optional[str] = None,
    user_features_transformed_path: Optional[str] = None,
    item_features_transformed_path: Optional[str] = None,
    user_item_features_path: Optional[str] = None 
    ) -> Tuple:
    """
    Loads recommender, user_features_transformed, item_features_transformed, 
    user_item_features for inference
    """
    if path is None:
        path = PATH

    if recommender_path is None:
        recommender_path = path + RECOMMENDER_PATH
    recommender = joblib.load(recommender_path)

    if user_features_transformed_path is None:
        user_features_transformed_path = path + USER_FEATURES_TRANSFORMED_PATH
    user_features_transformed = pd.read_csv(user_features_transformed_path)

    if item_features_transformed_path is None:
        item_features_transformed_path = path + ITEM_FEATURES_TRANSFORMED_PATH
    item_features_transformed = pd.read_csv(item_features_transformed_path)

    if user_item_features_path is None:
        user_item_features_path = path + USER_ITEM_FEATURES_PATH
    user_item_features = pd.read_csv(user_item_features_path)

    return (recommender, user_features_transformed, 
            item_features_transformed, user_item_features) 