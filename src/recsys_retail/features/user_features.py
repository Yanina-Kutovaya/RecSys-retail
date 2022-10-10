import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from pickle import dump

logger = logging.getLogger(__name__)

ORDINAL_FEATURES = [
    'age_desc', 'income_desc', 'homeowner_desc', 
    'household_size_desc', 'kid_category_desc'
]
ONEHOT_FEATURES = ['marital_status_code', 'hh_comp_desc']

AGE = ['19-24', '25-34', '35-44', '45-54', '55-64', '65+']
INCOME = ['Under 15K', '15-24K', '25-34K', '35-49K', '50-74K', '75-99K', 
          '100-124K', '125-149K', '150-174K', '175-199K', '200-249K', '250K+']
HOMEOWNER = ['Homeowner']
HOUSEHOLD_SIZE = ['1', '2', '3', '4', '5+']
KID_CATEGORY = ['None/Unknown', '1', '2', '3+']

MARITAL_STATUS = ['A', 'B', 'U']
HH_COMP =['Single Male', 'Single Female', '2 Adults No Kids',
          '1 Adult Kids', '2 Adults Kids']

ORD_CATEGORIES = [AGE, INCOME, HOMEOWNER, HOUSEHOLD_SIZE, KID_CATEGORY]
OH_CATEGORIES = [MARITAL_STATUS, HH_COMP]

__all__ = ['transform_user_features']

def transform_user_features(
    user_features: pd.DataFrame,
    ordinal_features=None,
    onehot_features=None,
    ord_categories=None,
    oh_categories=None
    ) -> pd.DataFrame:

    logging.info('Transforming user_features...')

    if ordinal_features is None:
        ordinal_features = ORDINAL_FEATURES
        ord_categories = ORD_CATEGORIES

    if onehot_features is None:
        onehot_features = ONEHOT_FEATURES
        oh_categories = OH_CATEGORIES

    ord_encoder = OrdinalEncoder(
        categories=ord_categories, 
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    oh_encoder = OneHotEncoder(
        categories=oh_categories, 
        handle_unknown='ignore', 
        sparse=False
    )
    user_transformer = make_column_transformer(
        (ord_encoder, ordinal_features),
        (oh_encoder, onehot_features)
    )
    X = user_transformer.fit_transform(user_features)
    user_id = user_features['user_id']
    cols = ordinal_features
    for i, col in enumerate(onehot_features):
        prefix = col + ' '
        cols +=  [prefix + cat for cat in oh_categories[i]]    

    dump(user_transformer, open('user_features_transformer_v1.pkl', 'wb'))
    
    return pd.DataFrame(X, index=user_id, columns=cols).reset_index()