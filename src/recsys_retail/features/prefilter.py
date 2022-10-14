import logging
import pandas as pd
import numpy as np
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ['prefilter_items']


LOWER_PRICE_THRESHOLD = 1
UPPER_PRICE_THRESHOLD = 30
TAKE_N_POPULAR = 2500

PATH = 'data/02_intermediate/'
PREFILTERED_TRAIN_DATA_LEVEL_1_PATH = PATH + 'data_train_lvl_1_prefiltered.csv.zip'

def prefilter_items(
    data: pd.DataFrame,
    item_features: pd.DataFrame,
    lower_price_threshold: Optional[int] = None,
    upper_price_threshold: Optional[int] = None, 
    take_n_popular: Optional[int] = None,
    prefilted_train_data_lvl_1_path: Optional[str] = None    
    ) -> pd.DataFrame:
    """ 
    1.Removes items that have not been sold for the last 12 months
    2.Removes most popular items (they will be bought anyway).
    3.Removes most unpopular items (nobody will buy them)
    4.Removes items from the departments with a limited assortiment
    5.Removes too cheap items (we will not earn on them)
    6.Removes too expensive_items. They will be bought irrespective
      of our recommendations.
    7.Selects top N popular items
    8.Introduces fake item_id = 999999. If user has bought an item which
      is not from top-N, he bought an item 999999.
    """

    logging.info('Prefiltering items...')

    # 1.Remove items that have not been sold for the last 12 months
    t = max(data['day']) - 365
    not_sold_in_12_month = list(
        set(data[data['day'] < t]['item_id'].unique()) -\
        set(data[data['day'] >= t]['item_id'].unique())
    )
    data = data[~data['item_id'].isin(not_sold_in_12_month)]

    # 2.Remove most popular items (they will be bought anyway).
    popularity = (
        data.groupby('item_id')['user_id'].nunique() / data['user_id'].nunique()
    ).reset_index()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    top_popular = popularity[
        popularity['share_unique_users'] > 0.2
    ].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # 3.Remove most unpopular items (nobody will buy them)
    top_unpopular = popularity[
        popularity['share_unique_users'] < 0.02
    ].item_id.tolist()
    data = data[~data['item_id'].isin(top_unpopular)]    

    # 4.Remove items from the departments with a limited assortiment
    if item_features is not None:
        department_size = (
            item_features.groupby('department')['item_id'].nunique()\
                .sort_values(ascending=False)
        ).reset_index()
        department_size.columns = ['department', 'n_items']

        unpopular_departments = department_size[
            department_size['n_items'] < 150
        ].department.tolist()
        items_in_unpopular_departments = item_features[
            item_features['department'].isin(unpopular_departments)
        ].item_id.unique().tolist()
        data = data[~data['item_id'].isin(items_in_unpopular_departments)]

    # 5.Remove too cheap items (we will not earn on them)
    data['price'] = data['sales_value'] / np.maximum(data['quantity'], 1)
    if lower_price_threshold is None:
        lower_price_threshold = LOWER_PRICE_THRESHOLD  
    cheap_items = data.loc[
        data['price'] < lower_price_threshold, 'item_id'
        ].unique().tolist()
    data = data[~data['item_id'].isin(cheap_items)]

    # 6.Remove too expensive_items. 
    # They will be bought irrespective of our recommendations.
    if upper_price_threshold is None:
        upper_price_threshold = UPPER_PRICE_THRESHOLD    
    expensive_items = data.loc[
        data['price'] > upper_price_threshold, 'item_id'
        ].unique().tolist()
    data = data[~data['item_id'].isin(expensive_items)]

    # 7.Select top N popular items
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    if take_n_popular is None:
        take_n_popular = TAKE_N_POPULAR
    top = popularity.sort_values(
        'n_sold', ascending=False
    ).head(take_n_popular).item_id.tolist()
    
    # Introduce fake item_id = 999999.
    # If user has bought an item which is not from top-N, he bought an item 999999.
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    if prefilted_train_data_lvl_1_path is None:
        prefilted_train_data_lvl_1_path = PREFILTERED_TRAIN_DATA_LEVEL_1_PATH
    data.to_csv(prefilted_train_data_lvl_1_path, index=False, compression='zip')

    return data