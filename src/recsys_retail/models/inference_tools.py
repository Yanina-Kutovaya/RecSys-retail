import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "recsys_retail"))

import pandas as pd
from .load_artifacts import load_inference_artifacts
from features.candidates_lvl_2 import get_candidates
from features.targets import get_targets_lvl_2

N_ITEMS = 100


def preprocess(user_ids, user_list=False) -> pd.DataFrame:
    """
    Preprocesses user_id  for inference

    """
    (
        data_valid,
        recommender,
        user_features_transformed,
        item_features_transformed,
        user_item_features,
    ) = load_inference_artifacts()

    if not user_list:
        user_ids = [user_ids]
    df = pd.DataFrame(user_ids, index=range(len(user_ids)), columns=["user_id"])

    users_inference = get_candidates(recommender, data_valid, df, n_items=N_ITEMS)
    train_dataset_lvl_2 = get_targets_lvl_2(
        users_inference,
        data_valid,
        item_features_transformed,
        user_features_transformed,
        user_item_features,
        n_items=N_ITEMS,
    )

    return train_dataset_lvl_2.drop("target", axis=1).fillna(0)
