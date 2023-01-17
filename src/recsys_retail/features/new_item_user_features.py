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
    X["hour"] = X["trans_time"] // 100
    X["weekday"] = X["day"] % 7

    X = encode_users(X)
    X = encode_items(X)
    X = encode_baskets(X)
    X = encode_store(X)
    X = encode_week_no(X)
    X = encode_weekday(X)
    X = encode_trans_time(X)

    ignore_list = [
        "basket_id",
        "day",
        "quantity",
        "store_id",
        "trans_time",
        "week_no",
        "hour",
        "weekday",
    ]
    cols = [i for i in X.columns if not i in ignore_list]
    user_item_features = X[cols]

    df1, df2 = get_embeddings(recommender)
    user_item_features = user_item_features.merge(df1, on=["item_id"])
    user_item_features = user_item_features.merge(df2, on=["user_id"])

    return user_item_features


def encode_users(
    data: pd.DataFrame,
) -> pd.DataFrame:

    logging.info("Generating user features...")

    df = pd.DataFrame()
    df["n_baskets_user"] = data.groupby(["user_id"])["basket_id"].count()
    df["n_items_user"] = data.groupby(["user_id"])["item_id"].count()
    df["sales_value_user"] = data.groupby(["user_id"])["sales_value"].sum()
    df1 = data.groupby(["user_id"])["sales_value", "quantity"].sum()
    df["avg_price_user"] = df1["sales_value"] / df1["quantity"]
    df["n_stores_user"] = data.groupby(["user_id"])["store_id"].count()
    df["retail_disc_user"] = data.groupby(["user_id"])["retail_disc"].sum()
    df["retail_disc_sales_ratio_user"] = df["retail_disc_user"] / df["sales_value_user"]
    df["coupon_disc_user"] = data.groupby(["user_id"])["coupon_disc"].sum()
    df["coupon_disc_sales_ratio_user"] = df["coupon_disc_user"] / df["sales_value_user"]
    df["coupon_match_disc_user"] = data.groupby(["user_id"])["coupon_match_disc"].sum()
    df["coupon_match_disc_sales_ratio_user"] = (
        df["coupon_match_disc_user"] / df["sales_value_user"]
    )
    df["n_week_nos_user"] = (
        data.groupby(["user_id", "week_no"])["week_no"]
        .count()
        .groupby(["user_id"])
        .count()
    )
    df["mean_visits_interval_user"] = (
        data.groupby("user_id")["day"].max() - data.groupby("user_id")["day"].min()
    ) / data.groupby("user_id")["day"].count()

    df["mean_basket_value_user"] = (
        data.groupby(["user_id", "basket_id"])["sales_value"]
        .sum()
        .groupby("user_id")["sales_value"]
        .mean()
    )
    df["n_items_baskets_mean_user"] = (
        data.groupby(["user_id", "basket_id"])["item_id"]
        .count()
        .groupby(["user_id"])["item_id"]
        .mean()
    )
    df["n_items_baskets_max_user"] = (
        data.groupby(["user_id", "basket_id"])["item_id"]
        .count()
        .groupby(["user_id"])["item_id"]
        .max()
    )
    df["n_items_baskets_std_user"] = (
        data.groupby(["user_id", "basket_id"])["item_id"]
        .count()
        .groupby(["user_id"])["item_id"]
        .std()
    )
    data = pd.merge(data, df, on="user_id", how="left")

    logging.info("Calculating median transactions time for pair user-item...")

    df = pd.DataFrame()
    df["median_trans_hour_user-item"] = data.groupby(["user_id", "item_id"])[
        "hour"
    ].median()
    df["median_trans_weekday_user-item"] = data.groupby(["user_id", "item_id"])[
        "weekday"
    ].median()
    data = pd.merge(data, df, on=["user_id", "item_id"], how="left")

    logging.info("Summarising user transactions by weekday...")

    df = pd.DataFrame()
    df["n_baskets_weekday_user"] = data.groupby(["user_id", "weekday"])[
        "basket_id"
    ].count()
    df["n_items_weekday_user"] = data.groupby(["user_id", "weekday"])["item_id"].count()
    df["sales_value_weekday_user"] = data.groupby(["user_id", "weekday"])[
        "sales_value"
    ].sum()
    df1 = data.groupby(["user_id", "weekday"])["sales_value", "quantity"].sum()
    df["avg_price_weekday_user"] = df1["sales_value"] / df1["quantity"]
    data = pd.merge(data, df, on=["user_id", "weekday"], how="left")

    logging.info("Summarising user transactions by transaction hour...")

    df = pd.DataFrame()
    df["n_baskets_trans_time_user"] = data.groupby(["user_id", "trans_time"])[
        "basket_id"
    ].count()
    df["n_items_trans_time_user"] = data.groupby(["user_id", "trans_time"])[
        "item_id"
    ].count()
    df["sales_value_trans_time_user"] = data.groupby(["user_id", "trans_time"])[
        "sales_value"
    ].sum()
    data = pd.merge(data, df, on=["user_id", "trans_time"], how="left")

    return data


def encode_items(
    data: pd.DataFrame,
) -> pd.DataFrame:

    logging.info("Generating item features...")

    df = pd.DataFrame()
    df["n_baskets_item"] = data.groupby(["item_id"])["basket_id"].count()
    df["n_users_item"] = data.groupby(["item_id"])["user_id"].count()
    df["sales_value_item"] = data.groupby(["item_id"])["sales_value"].sum()
    df["n_stores_item"] = data.groupby(["item_id"])["store_id"].count()
    df["retail_disc_item"] = data.groupby(["item_id"])["retail_disc"].sum()
    df["retail_disc_sales_ratio_item"] = df["retail_disc_item"] / df["sales_value_item"]
    df["coupon_disc_item"] = data.groupby(["item_id"])["coupon_disc"].sum()
    df["coupon_disc_sales_ratio_item"] = df["coupon_disc_item"] / df["sales_value_item"]
    df["coupon_match_disc_item"] = data.groupby(["item_id"])["coupon_match_disc"].sum()
    df["coupon_match_disc_sales_ratio_item"] = (
        df["coupon_match_disc_item"] / df["sales_value_item"]
    )
    df["n_week_nos_item"] = (
        data.groupby(["item_id", "week_no"])["week_no"]
        .count()
        .groupby(["item_id"])
        .count()
    )
    data = pd.merge(data, df, on="item_id", how="left")

    logging.info("Summarising item transactions by weekday...")

    df = pd.DataFrame()
    df["n_baskets_weekday_item"] = data.groupby(["item_id", "weekday"])[
        "basket_id"
    ].count()
    df["n_users_weekday_item"] = data.groupby(["item_id", "weekday"])["user_id"].count()
    df["sales_value_weekday_item"] = data.groupby(["item_id", "weekday"])[
        "sales_value"
    ].sum()
    df["n_stores_weekday_item"] = data.groupby(["item_id", "weekday"])[
        "store_id"
    ].count()
    data = pd.merge(data, df, on=["item_id", "weekday"], how="left")

    logging.info("Summarising item transactions by transaction hour...")

    df = pd.DataFrame()
    df["n_baskets_trans_time_item"] = data.groupby(["item_id", "trans_time"])[
        "basket_id"
    ].count()
    df["n_users_trans_time_item"] = data.groupby(["item_id", "trans_time"])[
        "user_id"
    ].count()
    df["sales_value_trans_time_item"] = data.groupby(["item_id", "trans_time"])[
        "sales_value"
    ].sum()
    df["n_stores_trans_time_item"] = data.groupby(["item_id", "trans_time"])[
        "store_id"
    ].count()
    data = pd.merge(data, df, on=["item_id", "trans_time"], how="left")

    return data


def encode_baskets(
    data: pd.DataFrame,
) -> pd.DataFrame:

    logging.info("Encoding baskets...")

    df = pd.DataFrame()
    df["n_items_busket"] = data.groupby(["basket_id"])["item_id"].count()
    df["sales_value_basket"] = data.groupby(["basket_id"])["sales_value"].sum()
    df1 = data.groupby(["basket_id"])["sales_value", "quantity"].sum()
    df["avg_price_basket"] = df1["sales_value"] / df1["quantity"]
    df["retail_disc_basket"] = data.groupby(["basket_id"])["retail_disc"].sum()
    df["retail_disc_sales_ratio_basket"] = (
        df["retail_disc_basket"] / df["sales_value_basket"]
    )
    df["coupon_disc_basket"] = data.groupby(["basket_id"])["coupon_disc"].sum()
    df["coupon_disc_sales_ratio_basket"] = (
        df["coupon_disc_basket"] / df["sales_value_basket"]
    )
    df["coupon_match_disc_basket"] = data.groupby(["basket_id"])[
        "coupon_match_disc"
    ].sum()
    df["coupon_match_disc_sales_ratio_basket"] = (
        df["coupon_match_disc_basket"] / df["sales_value_basket"]
    )
    data = pd.merge(data, df, on="basket_id", how="left")

    logging.info("Summarising baskets transactions by weekday...")

    df = pd.DataFrame()
    df["sales_value_weekday_basket"] = data.groupby(["basket_id", "weekday"])[
        "sales_value"
    ].sum()
    data = pd.merge(data, df, on=["basket_id", "weekday"], how="left")

    logging.info("Summarising baskets transactions by transaction hour...")

    df = pd.DataFrame()
    df["sales_value_trans_time_item"] = data.groupby(["basket_id", "trans_time"])[
        "sales_value"
    ].sum()
    data = pd.merge(data, df, on=["basket_id", "trans_time"], how="left")

    return data


def encode_store(
    data: pd.DataFrame,
) -> pd.DataFrame:

    logging.info("Encoding stores...")

    df = pd.DataFrame()
    df["n_users_store"] = (
        data.groupby(["store_id", "user_id"])["user_id"]
        .count()
        .groupby(["store_id"])
        .count()
    )
    df["mean_user_value_store"] = (
        data.groupby(["store_id", "user_id"])["sales_value"]
        .sum()
        .groupby(["store_id"])
        .mean()
    )
    df["n_items_store"] = (
        data.groupby(["store_id", "item_id"])["item_id"]
        .count()
        .groupby(["store_id"])
        .count()
    )
    df["mean_item_value_store"] = (
        data.groupby(["store_id", "item_id"])["sales_value"]
        .sum()
        .groupby(["store_id"])
        .mean()
    )
    df["n_baskets_store"] = (
        data.groupby(["store_id", "basket_id"])["basket_id"]
        .count()
        .groupby(["store_id"])
        .count()
    )
    df["mean_basket_value_store"] = (
        data.groupby(["store_id", "basket_id"])["sales_value"]
        .sum()
        .groupby(["store_id"])
        .mean()
    )
    df["sales_value_store"] = data.groupby(["store_id"])["sales_value"].sum()
    df1 = data.groupby(["store_id"])["sales_value", "quantity"].sum()
    df["avg_price_store"] = df1["sales_value"] / df1["quantity"]
    df["retail_disc_store"] = data.groupby(["store_id"])["retail_disc"].sum()
    df["retail_disc_sales_ratio_store"] = (
        df["retail_disc_store"] / df["sales_value_store"]
    )
    df["coupon_disc_store"] = data.groupby(["store_id"])["coupon_disc"].sum()
    df["coupon_disc_sales_ratio_store"] = (
        df["coupon_disc_store"] / df["sales_value_store"]
    )
    df["coupon_match_disc_store"] = data.groupby(["store_id"])[
        "coupon_match_disc"
    ].sum()
    df["coupon_match_disc_sales_ratio_store"] = (
        df["coupon_match_disc_store"] / df["sales_value_store"]
    )
    df["n_week_nos_store"] = (
        data.groupby(["store_id", "week_no"])["week_no"]
        .count()
        .groupby(["store_id"])
        .count()
    )
    data = pd.merge(data, df, on=["store_id"], how="left")

    logging.info("Summarising stores transactions by weekday...")

    df = pd.DataFrame()
    df["n_users_weekday_store"] = (
        data.groupby(["store_id", "weekday", "user_id"])["user_id"]
        .count()
        .groupby(["store_id", "weekday"])
        .count()
    )
    df["n_items_weekday_store"] = (
        data.groupby(["store_id", "weekday", "item_id"])["item_id"]
        .count()
        .groupby(["store_id", "weekday"])
        .count()
    )
    df["n_baskets_weekday_store"] = (
        data.groupby(["store_id", "weekday", "basket_id"])["basket_id"]
        .count()
        .groupby(["store_id", "weekday"])
        .count()
    )
    df["sales_value_weekday_store"] = data.groupby(["store_id", "weekday"])[
        "sales_value"
    ].sum()
    data = pd.merge(data, df, on=["store_id", "weekday"], how="left")

    logging.info("Summarising stores transactions by transaction hour...")

    df = pd.DataFrame()
    df["n_users_trans_time_store"] = (
        data.groupby(["store_id", "trans_time", "user_id"])["user_id"]
        .count()
        .groupby(["store_id", "trans_time"])
        .count()
    )
    df["n_items_trans_time_store"] = (
        data.groupby(["store_id", "trans_time", "item_id"])["item_id"]
        .count()
        .groupby(["store_id", "trans_time"])
        .count()
    )
    df["n_baskets_trans_time_store"] = (
        data.groupby(["store_id", "trans_time", "basket_id"])["basket_id"]
        .count()
        .groupby(["store_id", "trans_time"])
        .count()
    )
    df["sales_value_trans_time_store"] = data.groupby(["store_id", "trans_time"])[
        "sales_value"
    ].sum()
    data = pd.merge(data, df, on=["store_id", "trans_time"], how="left")

    return data


def encode_week_no(
    data: pd.DataFrame,
) -> pd.DataFrame:

    logging.info("Encoding week number...")

    df = pd.DataFrame()
    df["n_users_week_no"] = (
        data.groupby(["week_no", "user_id"])["user_id"]
        .count()
        .groupby("week_no")
        .count()
    )
    df["n_items_week_no"] = (
        data.groupby(["week_no", "item_id"])["item_id"]
        .count()
        .groupby("week_no")
        .count()
    )
    df["n_baskets_week_no"] = (
        data.groupby(["week_no", "basket_id"])["basket_id"]
        .count()
        .groupby("week_no")
        .count()
    )
    df["n_stores_week_no"] = (
        data.groupby(["week_no", "store_id"])["store_id"]
        .count()
        .groupby("week_no")
        .count()
    )
    df["sales_value_week_no"] = data.groupby(["week_no"])["sales_value"].sum()
    df1 = data.groupby(["week_no"])["sales_value", "quantity"].sum()
    df["avg_price_week_no"] = df1["sales_value"] / df1["quantity"]
    df["retail_disc_week_no"] = data.groupby(["week_no"])["retail_disc"].sum()
    df["retail_disc_sales_ratio_week_no"] = (
        df["retail_disc_week_no"] / df["sales_value_week_no"]
    )
    df["coupon_disc_week_no"] = data.groupby(["week_no"])["coupon_disc"].sum()
    df["coupon_disc_sales_ratio_week_no"] = (
        df["coupon_disc_week_no"] / df["sales_value_week_no"]
    )
    df["coupon_match_disc_week_no"] = data.groupby(["week_no"])[
        "coupon_match_disc"
    ].sum()
    df["coupon_match_disc_sales_ratio_week_no"] = (
        df["coupon_match_disc_week_no"] / df["sales_value_week_no"]
    )
    data = pd.merge(data, df, on=["week_no"], how="left")

    logging.info("Summarising transactions by week number and weekday...")

    df = pd.DataFrame()
    df["sales_value_weekday_week_no"] = data.groupby(["week_no", "weekday"])[
        "sales_value"
    ].sum()
    data = pd.merge(data, df, on=["week_no", "weekday"], how="left")

    logging.info("Summarising transactions by week number and transaction hour...")

    df = pd.DataFrame()
    df["sales_value_trans_time_store"] = data.groupby(["week_no", "trans_time"])[
        "sales_value"
    ].sum()
    data = pd.merge(data, df, on=["week_no", "trans_time"], how="left")

    return data


def encode_weekday(
    data: pd.DataFrame,
) -> pd.DataFrame:

    logging.info("Encoding weekday...")

    df = pd.DataFrame()
    df["n_baskets_weekday"] = (
        data.groupby(["weekday", "basket_id"])["basket_id"]
        .count()
        .groupby("weekday")
        .count()
    )
    df["sales_weekday"] = data.groupby(["weekday"])["sales_value"].sum()

    df["avg_basket_value_weekday"] = df["sales_weekday"] / df["n_baskets_weekday"]
    df1 = data.groupby(["weekday"])["sales_value", "quantity"].sum()
    df["avg_price_weekday"] = df1["sales_value"] / df1["quantity"]
    df["retail_disc_weekday"] = data.groupby(["weekday"])["retail_disc"].sum()
    df["retail_disc_sales_ratio_weekday"] = (
        df["retail_disc_weekday"] / df["sales_weekday"]
    )
    df["coupon_disc_weekday"] = data.groupby(["weekday"])["coupon_disc"].sum()
    df["coupon_disc_sales_ratio_weekday"] = (
        df["coupon_disc_weekday"] / df["sales_weekday"]
    )
    df["coupon_match_disc_weekday"] = data.groupby(["weekday"])[
        "coupon_match_disc"
    ].sum()
    df["coupon_match_disc_sales_ratio_weekday"] = (
        df["coupon_match_disc_weekday"] / df["sales_weekday"]
    )
    data = pd.merge(data, df, on=["weekday"], how="left")

    logging.info("Summarising weekday transactions by transaction hour...")

    df = pd.DataFrame()
    df["sales_value_trans_time_weekday"] = data.groupby(["weekday", "trans_time"])[
        "sales_value"
    ].sum()
    data = pd.merge(data, df, on=["weekday", "trans_time"], how="left")

    return data


def encode_trans_time(
    data: pd.DataFrame,
) -> pd.DataFrame:

    logging.info("Encoding transactions time...")

    df = pd.DataFrame()
    df["n_baskets_trans_time"] = (
        data.groupby(["trans_time", "basket_id"])["basket_id"]
        .count()
        .groupby("trans_time")
        .count()
    )
    df["sales_trans_time"] = data.groupby(["trans_time"])["sales_value"].sum()
    df["avg_basket_value_trans_time"] = (
        df["sales_trans_time"] / df["n_baskets_trans_time"]
    )
    df1 = data.groupby(["trans_time"])["sales_value", "quantity"].sum()
    df["avg_price_trans_time"] = df1["sales_value"] / df1["quantity"]
    df["retail_disc_trans_time"] = data.groupby(["trans_time"])["retail_disc"].sum()
    df["retail_disc_sales_ratio_trans_time"] = (
        df["retail_disc_trans_time"] / df["sales_trans_time"]
    )
    df["coupon_disc_trans_time"] = data.groupby(["trans_time"])["coupon_disc"].sum()
    df["coupon_disc_sales_ratio_trans_time"] = (
        df["coupon_disc_trans_time"] / df["sales_trans_time"]
    )
    df["coupon_match_disc_trans_time"] = data.groupby(["trans_time"])[
        "coupon_match_disc"
    ].sum()
    df["coupon_match_disc_sales_ratio_trans_time"] = (
        df["coupon_match_disc_trans_time"] / df["sales_trans_time"]
    )
    data = pd.merge(data, df, on=["trans_time"], how="left")

    return data


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
