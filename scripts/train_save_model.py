#!/usr/bin/env python
"""Train and save model for RecSys-retail"""

import sys
import logging
import argparse
import pandas as pd
import lightgbm as lgb
from typing import NoReturn, Optional

from src.recsys_retail.data.make_dataset import load_data
from src.recsys_retail.data.validation import train_test_split
from src.recsys_retail.models import train
from src.recsys_retail.models.serialize import store


logger = logging.getLogger()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-d',
        '--data_path',
        required=False,
        default=None,
        help='transactions dataset store path',
    )
    argparser.add_argument(
        '-d',
        '--item_features_path',
        required=False,
        default=None,
        help='item features dataset store path',
    )
    argparser.add_argument(
        '-d',
        '--user_features_path',
        required=False,
        default=None,
        help='user features dataset store path',
    )
    argparser.add_argument(
        '-o', '--output', required=True, 
        help='filename to store model; extention .txt will be added at saving the model'
    )
    argparser.add_argument(
        '-v', '--verbose', help='increase output verbosity', action='store_true'
    )
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info('Reading data...')

    data, item_features, user_features = load_data(
        args.data_path, args.item_features_path, args.user_features_path
    )
    logging.info('Preprocessing data...')
    train_dataset_lvl_2 = train.get_train_dataset_lvl_2(
        data, item_features, user_features
    )
    logging.info('Training the model...')
    train_store(train_dataset_lvl_2, args.output)


def train_store(dataset: pd.DataFrame, filename: str) -> NoReturn:
    """
    Trains and stores LightGBM model.
    Extention .txt will be added at saving the model.
    """
    
    X_train, X_valid, y_train, y_valid = train_test_split(dataset)
    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_valid, y_valid)

    logging.info(f'Training the model on {len(X_train)}  items...')

    params_lgb = {"boosting_type": "gbdt",
                  "objective": "binary", 
                  "metric": "auc",
                  "num_boost_round": 10000,
                  "learning_rate": 0.01,
                  "class_weight": 'balanced',
                  "max_depth": 10,
                  "n_estimators": 5000,
                  "n_jobs": 6,
                  "seed": 12}    

    model_lgb = lgb.train(
        params=params_lgb, train_set=dtrain, valid_sets=[dtrain, dvalid],
        verbose_eval=1000, early_stopping_rounds=30
    )          
    store(model_lgb, filename)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e)
        sys.exit(1)
