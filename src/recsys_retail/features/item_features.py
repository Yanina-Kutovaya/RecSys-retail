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

PATH = 'data/02_intermediate/'
FREQUENCY_ENCODER_PATH = PATH + 'item_features_frequency_encoder_v1.pkl'
ITEM_FEATURES_FREQ_ENCODED_PATH = PATH + 'item_features_frequency_encoded.csv.zip'

VECTORIZER_PATH = PATH + 'item_features_vectorizer_v1.pkl'
ITEM_DESC_VECTORIZED_PATH = PATH + 'item_desc_vectorized.csv.zip'

ITEM_FEATURES_TRANSFORMED_PATH = PATH + 'item_features_transformed.csv.zip'
ITEM_FEATURES_FOR_INFERENCE_PATH = PATH + 'item_features_for_inference.csv.zip'


def item_features_frequency_encoder(
    item_features: pd.DataFrame,
    frequency_encoder_path: Optional[str] = None,
    item_features_freq_encoded_path: Optional[str] = None
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
    if frequency_encoder_path is None:
        frequency_encoder_path = FREQUENCY_ENCODER_PATH
    dump(frequency_encoder, open(frequency_encoder_path, 'wb'))

    X = pd.DataFrame(X, index=item_id, columns=cols).reset_index()
    if item_features_freq_encoded_path is None:
        item_features_freq_encoded_path = ITEM_FEATURES_FREQ_ENCODED_PATH
    X.to_csv(item_features_freq_encoded_path, index=False, compression='zip')
    
    item_features.reset_index(inplace=True)

    return X


def item_features_descriptions_encoder(
    item_features: pd.DataFrame,
    add_stop_words=None,
    concat_list=None,
    vectorizer_path: Optional[str] = None,
    item_desc_vectorized_path: Optional[str] = None   
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
    if vectorizer_path is None:
        vectorizer_path = VECTORIZER_PATH
    dump(vectorizer, open(vectorizer_path, 'wb')) 

    X = pd.DataFrame(X.toarray(), index=item_features['item_id']).reset_index()
    if item_desc_vectorized_path is None:
        item_desc_vectorized_path = ITEM_DESC_VECTORIZED_PATH    
    X.to_csv(item_desc_vectorized_path, index=False, compression='zip')       

    return X

    
def fit_transform_item_features(
    item_features: pd.DataFrame,
    item_features_transformed_path: Optional[str] = None
    ) -> pd.DataFrame:
    """
    Combines frequency encoding and item descriptions encoding 
    for training of the models.
    """
    logging.info('Transforming item_features for train dataset...')

    df1 = item_features_frequency_encoder(item_features)
    df2 = item_features_descriptions_encoder(item_features)

    X = pd.merge(df1, df2, on='item_id', how='left')    
    if item_features_transformed_path is None:
        item_features_transformed_path = ITEM_FEATURES_TRANSFORMED_PATH
    X.to_csv(item_features_transformed_path, index=False, compression='zip')

    return X


def transform_item_features(
    item_features: pd.DataFrame,
    frequency_encoder_path = None,
    concat_list = None,
    vectorizer_path: Optional[str] = None,    
    item_features_for_inference_path: Optional[str] = None
    ) -> pd.DataFrame:

    """
    Combines frequency encoding and item descriptions encoding 
    for inference.
    """

    logging.info('Transforming item_features for inference...')

    if frequency_encoder_path is None:
        frequency_encoder_path = FREQUENCY_ENCODER_PATH
    frequency_encoder = load(open(frequency_encoder_path, 'rb'))
    df1 = frequency_encoder.transform(item_features)

    if concat_list is None:
        concat_list = CONCAT_LIST
    item_features.loc[:, 'item_desc'] = item_features.loc[:, concat_list].apply(
        lambda x: ' '.join(x).replace("/", " ").replace("-", " "), axis=1
    )
    if vectorizer_path is None:
        vectorizer_path = VECTORIZER_PATH
    vectorizer = load(open(vectorizer_path, 'rb'))
    df2 = vectorizer.transform(item_features['item_desc'])

    X = pd.merge(df1, df2, on='item_id', how='left')
    if item_features_for_inference_path is None:
        item_features_for_inference_path = ITEM_FEATURES_FOR_INFERENCE_PATH
    X.to_csv(item_features_for_inference_path, index=False, compression='zip')

    return X