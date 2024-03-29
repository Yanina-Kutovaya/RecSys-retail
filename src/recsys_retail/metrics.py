import joblib
import numpy as np
import pandas as pd
from typing import Optional

FOLDER = "data/04_feature/"
PREFILTERED_ITEM_LIST_PATH = FOLDER + "prefiltered_item_list.joblib"


def get_recommendations(
    train_dataset_lvl_2: pd.DataFrame, raw_predictions, k: int = 5
) -> pd.DataFrame:
    """
    Transforms model predictions to recommendations

    train_dataset_lvl_2 - dataset used for inference
    raw_predictions - predictions generated by the model
    k - the number of items to be recommended
    """

    df = train_dataset_lvl_2[["user_id", "item_id"]]
    df["predictions"] = raw_predictions
    df = df.groupby(["user_id", "item_id"])["predictions"].median().reset_index()
    df = df.sort_values(["predictions"], ascending=False).groupby(["user_id"]).head(k)
    recommendations = df.groupby("user_id")["item_id"].unique().reset_index()
    recommendations.columns = ["user_id", "recommendations"]

    return recommendations


def get_results(
    data_val_lvl_2: pd.DataFrame,
    train_dataset_lvl_2: pd.DataFrame,
    raw_predictions,
    prefiltered_item_list_path: Optional[str] = None,
    k: int = 5,
) -> pd.DataFrame:
    """
    Compares recommendations against actual purchases

    actual - items actually bought by the users
    actual_adj	- items bought by the user from prefiltered list
        (from train dataset)
    recommendations - list of items recommended for each user
        (generated by the model)

    k: the number of items for calculation of precision@k
    """

    if prefiltered_item_list_path is None:
        prefiltered_item_list_path = PREFILTERED_ITEM_LIST_PATH

    recomendations = get_recommendations(train_dataset_lvl_2, raw_predictions, k)

    actuals = data_val_lvl_2.groupby("user_id")["item_id"].unique().reset_index()
    actuals.columns = ["user_id", "actual"]

    test_list = data_val_lvl_2["item_id"].unique()

    prefiltered_item_list = joblib.load(prefiltered_item_list_path)
    rec_used = list(set(test_list) & set(prefiltered_item_list))

    actuals_adj = (
        data_val_lvl_2[data_val_lvl_2["item_id"].isin(rec_used)]
        .groupby("user_id")["item_id"]
        .unique()
        .reset_index()
    )
    actuals_adj.columns = ["user_id", "actual_adj"]
    actuals = actuals.merge(actuals_adj, on="user_id", how="left")

    results = actuals.merge(recomendations, on="user_id", how="left")

    return results


def adjust_results_for_metrics(results: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Selects users who bought k and more items from prefiltered list (items from
    train dataset) ti be used in metrics calculations.

    k: the number of items for calculation of precision@k
    results: the output of the function get_results

    actual: list of items actually bought by the users
    actual_adj: items bought by the user from prefiltered list
        (from train dataset)
    recommendations: list of items recommended for each user
        (generated by the model)
    len_actual_adj: the number of items from prefiltered list bought by the user.
    """

    results = results[~results["actual_adj"].isnull()]

    results["len_actual_adj"] = 0
    for i in range(len(results)):
        results.iloc[i, 4] = len(results.iloc[i, 2])

    results = results[results["len_actual_adj"] >= k]

    return results


def precision_at_k(recommended_list: list, bought_list: list, k: int = 5):
    """
    Calculates precision@k

    recommended_list - the list of items recommended for the user
    bought_list - the list of items actually bought by the user
    k - the number of items for calculation of precision@k
    """

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    return len(set(bought_list) & set(recommended_list)) / len(recommended_list)
