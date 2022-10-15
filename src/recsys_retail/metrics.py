import numpy as np
import pandas as pd


def get_results(
    train_dataset_lvl_2: pd.DataFrame, 
    raw_predictions: np.array,
    ) -> pd.DataFrame:
    """
    Compares recommendations against actual purchases
    """

    df = train_dataset_lvl_2[['user_id', 'item_id']]
    df['predictions'] = raw_predictions

    df = df.groupby(['user_id', 'item_id'])['predictions'].median().reset_index()
    df = df.sort_values(['predictions'], ascending=False).groupby(['user_id']).head(5)

    recomendations = df.groupby('user_id')['item_id'].unique().reset_index()
    recomendations.columns = ['user_id', 'recommendations']

    actuals = data_val_lvl_2.groupby('user_id')['item_id'].unique().reset_index()
    actuals.columns=['user_id', 'actual']
    
    return actuals.merge(recomendations, on='user_id', how='left')


def precision_at_k(recommended_list, bought_list, k=5) -> np.float:
    """
    Calculates precision@k
    """

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)
    precision = flags.sum() / len(recommended_list)

    return precision