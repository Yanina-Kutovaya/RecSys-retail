#!/usr/bin/env python3
"""Train and save model for RecSys-retail"""

import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

import logging
import argparse
import pandas as pd
from catboost import CatBoostClassifier, Pool
from typing import Optional

from src.recsys_retail.data.make_dataset import load_data
from src.recsys_retail.data.validation import train_test_split
from src.recsys_retail.models import train
from src.recsys_retail.models.serialize import store


logger = logging.getLogger()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-d1",
        "--data_path",
        required=False,
        default="https://storage.yandexcloud.net/recsys-retail-input/train.csv.zip",
        help="transactions dataset store path",
    )
    argparser.add_argument(
        "-d2",
        "--item_features_path",
        required=False,
        default="https://storage.yandexcloud.net/recsys-retail-input/item_features.csv",
        help="item features dataset store path",
    )
    argparser.add_argument(
        "-d3",
        "--user_features_path",
        required=False,
        default="https://storage.yandexcloud.net/recsys-retail-input/user_features.csv",
        help="user features dataset store path",
    )
    argparser.add_argument(
        "-o",
        "--output",
        required=True,
        help="filename to store model",
    )
    argparser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Reading data...")

    data, item_features, user_features = load_data(
        args.data_path, args.item_features_path, args.user_features_path
    )
    logging.info("Preprocessing data...")
    train_dataset_lvl_2 = train.data_preprocessing_pipeline(
        data, item_features, user_features
    )
    logging.info("Training the model...")
    train_store(train_dataset_lvl_2, args.output)


def train_store(dataset: pd.DataFrame, filename: str):
    """
    Trains and stores CatBoost model.
    """

    X_train, X_valid, y_train, y_valid = train_test_split(dataset)
    train_data = Pool(X_train, y_train)
    eval_data = Pool(X_valid, y_valid)

    logging.info(f"Training the model on {len(X_train)}  items...")

    model_cb = CatBoostClassifier(
        learning_rate=0.005, early_stopping_rounds=20, eval_metric="AUC", random_seed=42
    )
    model_cb.fit(train_data, eval_set=eval_data, verbose=50)
    store(model_cb, filename)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e)
        sys.exit(1)
