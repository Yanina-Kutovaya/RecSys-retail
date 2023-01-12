import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["load_train_dataset"]

PATH = "data/01_raw/"

TRAIN_PATH = PATH + "train.csv"
ITEM_FEATURES_PATH = PATH + "item_features.csv"
USER_FEATURES_PATH = PATH + "user_features.csv"


def load_data(
    data_path: Optional[str] = None,
    item_path: Optional[str] = None,
    user_path: Optional[str] = None,
) -> pd.DataFrame:

    if data_path is None:
        data_path = TRAIN_PATH
    logging.info(f"Reading dataset from {data_path}...")
    data = pd.read_csv(data_path)

    if item_path is None:
        item_path = ITEM_FEATURES_PATH
    logging.info(f"Reading item_features from {item_path}...")
    item_features = pd.read_csv(item_path)
    item_features.columns = map(str.lower, item_features.columns)
    item_features.rename(columns={"product_id": "item_id"}, inplace=True)

    if user_path is None:
        user_path = USER_FEATURES_PATH
    logging.info(f"Reading user_features from {user_path}...")
    user_features = pd.read_csv(user_path)
    user_features.columns = map(str.lower, user_features.columns)
    user_features.rename(columns={"household_key": "user_id"}, inplace=True)

    return data, item_features, user_features
