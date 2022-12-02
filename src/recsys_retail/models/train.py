import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "recsys_retail"))

import logging
import numpy as np
import pandas as pd
from pickle import dump
from typing import Optional

from data.make_dataset import load_data
from features.data_time_split import time_split_2
from features.prefilter import prefilter_items
from features.user_features import fit_transform_user_features
from features.item_features import fit_transform_item_features
from features.recommenders import MainRecommender
from features.candidates_lvl_2 import get_candidates
from features.new_item_user_features import get_user_item_features
from features.targets import get_targets_lvl_2
from .save_artifacts import (
    save_time_split,
    save_prefiltered_data,
    save_item_featutes,
    save_user_features,
    save_preprocessed_lvl_1_train_dataset,
    save_recommender,
    save_candidates,
    save_user_item_features,
    save_train_dataset_lvl_2,
)

logger = logging.getLogger(__name__)

__all__ = ["preprocess_data"]

N_FACTORS_ALS = 50
N_ITEMS = 100


def data_preprocessing_pipeline(
    data: pd.DataFrame,
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    n_factors_ALS: Optional[int] = None,
    n_items: Optional[int] = None,
    save_artifacts=True,
) -> pd.DataFrame:

    """
    Prepares dataset from transactions data, item features and user features
    to be used in binary classification models.

    n_items - the number of items selected by the recommender for each user
              on the 1st stage (long list), the bases for the further short list
              selection with the binary classification model.

    """

    logging.info("Prefiltering transactions data...")
    data = prefilter_items(data, item_features)

    logging.info("Splitting data on train and validation datasets...")
    data_train, data_valid = time_split_2(data)

    logging.info("Preprocessing train dataset...")

    user_features_transformed = fit_transform_user_features(user_features)
    data_train = pd.merge(
        data_train, user_features_transformed, on="user_id", how="left"
    )
    item_features_transformed = fit_transform_item_features(item_features)
    data_train = pd.merge(
        data_train, item_features_transformed, on="item_id", how="left"
    )
    logging.info("Training recommender...")

    if n_factors_ALS is None:
        n_factors_ALS = N_FACTORS_ALS

    recommender = MainRecommender(
        data_train,
        n_factors_ALS=n_factors_ALS,
        regularization_ALS=0.001,
        iterations_ALS=15,
        num_threads_ALS=4,
    )

    logging.info("Selecting users for level 2 dataset...")

    if n_items is None:
        n_items = N_ITEMS

    users_lvl_2 = get_candidates(recommender, data_train, data_valid, n_items=n_items)
    logging.info("Generating new user-item features for level 2 model...")

    user_item_features = get_user_item_features(recommender, data_train)

    logging.info("Generating train dataset for level 2 model...")

    train_dataset_lvl_2 = get_targets_lvl_2(
        users_lvl_2,
        data_valid,
        item_features_transformed,
        user_features_transformed,
        user_item_features,
        n_items,
    )

    if save_artifacts:
        save_prefiltered_data(data)
        save_time_split(data_train, data_valid)
        save_item_featutes(item_features_transformed)
        save_user_features(user_features_transformed)
        save_preprocessed_lvl_1_train_dataset(data_train)
        save_recommender(recommender)
        save_candidates(users_lvl_2)
        save_user_item_features(user_item_features)
        save_train_dataset_lvl_2(train_dataset_lvl_2)

    return train_dataset_lvl_2
