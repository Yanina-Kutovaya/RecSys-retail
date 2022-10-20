import os
import logging
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from pickle import dump, load
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ['transform_item_features']


ADD_STOP_WORDS = [
    '10', '100', '100 pure', '12', '12 18',
    '15pk', '15pk can', '18','18 15pk', '50'
]
CONCAT_LIST = ['commodity_desc', 'sub_commodity_desc']


def item_features_frequency_encoder(
    item_features: pd.DataFrame    
    ) -> pd.DataFrame:
    """
    Assumes all the features as categorical and replaces categories
    with their frequencies.
    """

    logging.info('Encoding item features with frequencies...')

    item_features.set_index('item_id', inplace=True)
    cols = item_features.columns
    item_id = item_features.index

    frequency_encoder = make_pipeline(
        ce.count.CountEncoder(cols=cols, normalize=True),
        MinMaxScaler()
    )
    X = frequency_encoder.fit_transform(item_features)
    item_features_frequency_encoded = pd.DataFrame(
        X, index=item_id, columns=cols
    ).reset_index()    
    
    item_features.reset_index(inplace=True)

    return item_features_frequency_encoded


def item_features_descriptions_encoder(
    item_features: pd.DataFrame,
    add_stop_words=None,
    concat_list=None,
      
    )-> pd.DataFrame:
    """
    Encodes commodity and sub-commodity descriptions with TF-IDF vectorizer.
    """

    logging.info('Encoding item features descriptions...')    

    if concat_list is None:
        concat_list = CONCAT_LIST
    item_features.loc[:, 'item_desc'] = item_features.loc[:, concat_list].apply(
        lambda x: ' '.join(x).replace("/", " ").replace("-", " "), axis=1
    )
    if add_stop_words is None:
        add_stop_words = ADD_STOP_WORDS
    my_stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), analyzer='word', lowercase=True, 
        max_features=500, stop_words=my_stop_words
    )
    X = vectorizer.fit_transform(item_features['item_desc'])
    item_desc_vectorized = pd.DataFrame(
        X.toarray(), index=item_features['item_id']
    ).reset_index()           

    return item_desc_vectorized

    
def fit_transform_item_features(
    item_features: pd.DataFrame,
    vectorize_desc = False
    ) -> pd.DataFrame:
    """
    Combines frequency encoding and item descriptions encoding 
    for training of the models.
    """
    logging.info('Transforming item_features for train dataset...')

    df1 = item_features_frequency_encoder(item_features)

    if vectorize_desc:
        df2 = item_features_descriptions_encoder(item_features)
        item_features_transformed = pd.merge(df1, df2, on='item_id', how='left')
    else:
        item_features_transformed = df1

    return item_features_transformed 