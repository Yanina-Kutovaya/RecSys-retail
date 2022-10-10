import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ['transform_user_features']

def transform_user_features(user_features: pd.DataFrame)-> pd.DataFrame:

    logging.info('Transforming user_features...')

    user_features['age_desc'].replace(
    {'19-24': 22, '25-34': 30, '35-44': 40, '45-54': 50, '55-64': 60, '65+': 70},
    inplace=True
    )
    user_features['marital_status_code'].replace(
        {'U': 0, 'A': 1, 'B': 2}, inplace=True
    )
    user_features['income_desc'].replace(
        {'Under 15K': 10, '15-24K': 20, '25-34K':30, '35-49K': 40, 
        '50-74K': 62, '75-99K': 87, '100-124K': 112, '125-149K': 137, 
        '150-174K': 162, '175-199K': 187, '200-249K': 225, '250K+':275}, inplace=True
    )
    user_features['homeowner_desc'] = np.where(
        user_features['homeowner_desc']=='Homeowner', 1, 0
    )
    user_features['hh_comp_desc'].replace(
        {'Unknown': 0, 'Single Male': 1, 'Single Female': 2,
        '1 Adult Kids': 3, '2 Adults No Kids': 4, '2 Adults Kids':5}, inplace=True
    )
    user_features['household_size_desc'].replace({'5+': 5}, inplace=True)
    user_features['kid_category_desc'].replace(
        {'None/Unknown': 0, '3+': 3}, inplace=True
    )

    return user_features