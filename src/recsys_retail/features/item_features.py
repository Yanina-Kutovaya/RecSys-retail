import logging
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from pickle import dump

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

    logging.info('Encoding item features frequencies...')

    item_features.set_index('item_id', inplace=True)
    cols = item_features.columns
    item_id = item_features.index

    frequency_encoder = make_pipeline(
        ce.count.CountEncoder(cols=cols, normalize=True),
        MinMaxScaler()
    )
    X = frequency_encoder.fit_transform(item_features)    
    dump(frequency_encoder, open('item_features_frequency_encoder_v1.pkl', 'wb'))
    item_features.reset_index(inplace=True)

    return pd.DataFrame(X, index=item_id, columns=cols).reset_index()


def item_features_descriptions_encoder(
    item_features: pd.DataFrame,
    add_stop_words=None,
    concat_list=None    
    )-> pd.DataFrame:

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
    dump(vectorizer, open('item_features_vectorizer_v1.pkl', 'wb'))    

    return pd.DataFrame(X.toarray(), index=item_features['item_id']).reset_index()


def transform_item_features(item_features: pd.DataFrame) -> pd.DataFrame:

    logging.info('Transforming item_features...')

    df1 = item_features_frequency_encoder(item_features)
    df2 = item_features_descriptions_encoder(item_features)

    return pd.merge(df1, df2, on='item_id', how='left')
