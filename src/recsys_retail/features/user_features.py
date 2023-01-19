import logging
import numpy as np
import pandas as pd
import category_encoders as ce
from pickle import dump, load
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["transform_user_features"]


OH_FEATURES = ["marital_status_code", "hh_comp_desc", "homeowner_desc"]
ORDINAL_FEATURES = [
    "age_desc",
    "income_desc",
    "household_size_desc",
    "kid_category_desc",
]
AGE = ["19-24", "25-34", "35-44", "45-54", "55-64", "65+"]
INCOME = [
    "Under 15K",
    "15-24K",
    "25-34K",
    "35-49K",
    "50-74K",
    "75-99K",
    "100-124K",
    "125-149K",
    "150-174K",
    "175-199K",
    "200-249K",
    "250K+",
]
HOUSEHOLD_SIZE = ["1", "2", "3", "4", "5+"]
KID_CATEGORY = ["None/Unknown", "1", "2", "3+"]

FEATRURE_CATEGORY_MAPPING = {
    "age_desc": AGE,
    "income_desc": INCOME,
    "household_size_desc": HOUSEHOLD_SIZE,
    "kid_category_desc": KID_CATEGORY,
}


def fit_transform_user_features(
    user_features: pd.DataFrame,
    onehot_features=None,
    ordinal_features=None,
) -> pd.DataFrame:

    """
    Encodes categorical features with OneHotEncoder, OrdinalEncoder and HelmertEncoder.

    """

    logging.info("Transforming user_features...")

    if onehot_features is None:
        onehot_features = OH_FEATURES

    if ordinal_features is None:
        ordinal_features = ORDINAL_FEATURES

    user_features.set_index("user_id", inplace=True)

    encoder_oh = ce.OneHotEncoder(
        cols=onehot_features,
        drop_invariant=True,
        return_df=True,
        handle_missing="return_nan",
        handle_unknown=-1,
        use_cat_names=True,
    )
    col_encodings = get_ordinal_encodings()
    encoder_ord = ce.OrdinalEncoder(
        mapping=col_encodings,
        cols=ordinal_features,
        drop_invariant=True,
        handle_unknown=-1,
        handle_missing=-2,
    )
    encoder_helmert = ce.HelmertEncoder(
        mapping=col_encodings,
        cols=ordinal_features,
        drop_invariant=True,
        return_df=True,
        handle_unknown=-1,
        handle_missing=-2,
    )

    df1 = encoder_oh.fit_transform(user_features)
    cols1 = [i for i in encoder_oh.feature_names if not i in ordinal_features]

    df2 = encoder_ord.fit_transform(user_features)
    cols2 = [i for i in encoder_ord.feature_names if not i in onehot_features]

    df3 = encoder_helmert.fit_transform(user_features)
    cols3 = [i for i in encoder_helmert.feature_names if not i in onehot_features]

    user_features_transformed = pd.concat(
        [df1[cols1], df2[cols2], df3[cols3]], axis=1
    ).reset_index()

    return user_features_transformed


def get_ordinal_encodings(feature_category_mapping=None):
    """
    Generates feature-category mapping for OrdinalEncoder and HelmertEncoder

    """
    if feature_category_mapping is None:
        feature_category_mapping = FEATRURE_CATEGORY_MAPPING

    col_encodings = []
    for col in feature_category_mapping.keys():
        col_encoding = {}
        col_encoding["col"] = col
        map_dict = {}
        for i, item in enumerate(feature_category_mapping[col]):
            map_dict[item] = i
            col_encoding["mapping"] = map_dict
            col_encodings.append(col_encoding)

    return col_encodings
