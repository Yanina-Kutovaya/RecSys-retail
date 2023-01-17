import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["generate_user_item_features"]


def get_user_item_features(
    recommender,
    data_train_lvl_1: pd.DataFrame,
) -> pd.DataFrame:

    """
    Generates new features from transactions matrix, users and items
    embeddings taken from recommender.

    """

    logging.info("Generating new user-item features...")

    X = data_train_lvl_1.copy()
    user_item_features = X[["user_id", "item_id"]]

    df = pd.DataFrame()
    df["median_trans_time"] = X.groupby(["user_id", "item_id"])["trans_time"].median()
    df["median_trans_weekday"] = X.groupby(["user_id", "item_id"])["weekday"].median()
    user_item_features = pd.merge(
        user_item_features, df, on=["user_id", "item_id"], how="left"
    )
    df = pd.DataFrame()
    df["mean_visits_interval"] = (
        X.groupby("user_id")["day"].max() - X.groupby("user_id")["day"].min()
    ) / X.groupby("user_id")["day"].nunique()

    df["n_baskets_trans_time_min"] = (
        X.groupby(["user_id", "trans_time"])["n_baskets_trans_time"]
        .count()
        .groupby(["user_id"])
        .min()
    )
    df["n_baskets_trans_time_max"] = (
        X.groupby(["user_id", "trans_time"])["n_baskets_trans_time"]
        .count()
        .groupby(["user_id"])
        .max()
    )
    df["n_baskets_trans_time_var"] = (
        X.groupby(["user_id", "trans_time"])["n_baskets_trans_time"]
        .count()
        .groupby(["user_id"])
        .std()
    )
    user_item_features = pd.merge(user_item_features, df, on=["user_id"], how="left")

    df1, df2 = get_embeddings(recommender)
    user_item_features = user_item_features.merge(df1, on=["item_id"])
    user_item_features = user_item_features.merge(df2, on=["user_id"])

    return user_item_features


def get_embeddings(recommender) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates embeddings from recommender item and user factors.
    """

    logging.info("Calculating item embeddings...")

    df1 = recommender.model.item_factors
    if not type(df1) is np.ndarray:
        df1 = df1.to_numpy()
    n_factors = df1.shape[1]
    ind = list(recommender.id_to_itemid.values())
    df1 = pd.DataFrame(df1, index=ind).reset_index()
    df1.columns = ["item_id"] + ["factor_" + str(i + 1) for i in range(n_factors)]

    logging.info("Calculating user embeddings...")

    df2 = recommender.model.user_factors
    if not type(df2) is np.ndarray:
        df2 = df2.to_numpy()
    n_users = df2.shape[0]
    ind = list(recommender.id_to_userid.values())[:n_users]
    df2 = pd.DataFrame(df2, index=ind).reset_index()
    df2.columns = ["user_id"] + ["user_factor_" + str(i + 1) for i in range(n_factors)]

    return df1, df2
