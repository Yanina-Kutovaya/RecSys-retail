import os
import logging
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from pickle import dump, load
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["transform_item_features"]


FEATURES_FOR_COUNT_ENCODER = [
    "manufacturer",
    "department",
    "commodity_desc",
    "sub_commodity_desc",
    "curr_size_of_product",
]
FEATURE_FOR_HASHING_ENCODER = "manufacturer"

ADD_STOP_WORDS = [
    "10",
    "100",
    "100 pure",
    "12",
    "12 18",
    "15pk",
    "15pk can",
    "18",
    "18 15pk",
    "50",
]
CONCAT_LIST = ["commodity_desc", "sub_commodity_desc"]


def fit_transform_item_features(
    item_features: pd.DataFrame,
    count_cols=None,
    hashing_enc_col=None,
) -> pd.DataFrame:
    """
    Generates new item featurs for train dataset.
    """
    logging.info("Transforming item_features for train dataset...")

    if count_cols is None:
        count_cols = FEATURES_FOR_COUNT_ENCODER
    if hashing_enc_col is None:
        hashing_enc_col = FEATURE_FOR_HASHING_ENCODER

    item_features.set_index("item_id", inplace=True)

    logging.info("Encoding feature 'brand' ...")

    df1 = pd.DataFrame()
    df1["brand"] = item_features["brand"].map({"Private": 0, "National": 1})

    logging.info("Encoding item features with CountEncoder ...")

    count_encoder = ce.CountEncoder(
        cols=count_cols,
        handle_unknown=-1,
        handle_missing=-2,
        min_group_size=5,
        combine_min_nan_groups=True,
        min_group_name="others",
        normalize=True,
    )
    df2 = count_encoder.fit_transform(item_features[count_cols])
    df2.columns = [i + "_count" for i in df2.columns]

    logging.info("Encoding features with HashingEncoder ...")

    hashing_encoder = ce.HashingEncoder(cols=[hashing_enc_col])
    df3 = hashing_encoder.fit_transform(item_features[[hashing_enc_col]])
    df3.columns = [hashing_enc_col + "_" + str(i) for i in range(df3.shape[1])]

    logging.info("Encoding item descriptions...")

    df4 = encode_item_descriptions(item_features)

    item_features_transformed = pd.concat([df1, df2, df3, df4], axis=1)

    return item_features_transformed


def encode_item_descriptions(
    item_features: pd.DataFrame,
    add_stop_words=None,
    concat_list=None,
) -> pd.DataFrame:
    """
    Encodes commodity and sub-commodity descriptions with TF-IDF vectorizer and applies
    HashingEncoder to change the number od columns from 300 to 32.
    """

    if concat_list is None:
        concat_list = CONCAT_LIST
    if add_stop_words is None:
        add_stop_words = ADD_STOP_WORDS

    my_stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer="word",
        lowercase=True,
        max_features=300,
        stop_words=my_stop_words,
    )
    df = pd.DataFrame()
    df["item_desc"] = item_features.loc[:, concat_list].apply(
        lambda x: " ".join(x).replace("/", " ").replace("-", " "), axis=1
    )
    X = vectorizer.fit_transform(df["item_desc"])
    df_desc = X.toarray()
    df_desc = pd.DataFrame(df_desc, index=item_features.index)

    hashing_encoder = ce.HashingEncoder(cols=df_desc.columns, n_components=32)
    item_desc_encoded = hashing_encoder.fit_transform(df_desc)
    item_desc_encoded.columns = [
        "item_desc" + "_" + str(i) for i in range(item_desc_encoded.shape[1])
    ]

    return item_desc_encoded
