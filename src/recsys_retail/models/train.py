import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pickle import dump
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ['build_inference_pipeline']

PATH = 'models/'
MODEL_LGB_PATH = PATH + 'LightGBM_v1.pkl'

def run_model_lgb(
    targets_lvl_2: pd.DataFrame,
    model_lgb_path: Optional[str] = None
    ):

    logging.info('Training the model LightGBM...')

    X_train, X_valid, y_train, y_valid = train_test_split(
        targets_lvl_2.drop('target', axis=1).fillna(0), 
        targets_lvl_2[['target']], 
        test_size=0.2,
        random_state=16, 
        stratify=targets_lvl_2[['target']]
    )

    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_valid, y_valid)

    params_lgb = {"boosting_type": "gbdt",
                  "objective": "binary", 
                  "metric": "auc",
                  "num_boost_round": 10000,
                  "learning_rate": 0.1,
                  "class_weight": 'balanced',
                  "max_depth": 10,
                  "n_estimators": 5000,
                  "n_jobs": 6,
                  "seed": 12} 

    model_lgb = lgb.train(
        params=params_lgb, train_set=dtrain, valid_sets=[dtrain, dvalid],
        verbose_eval=1000, early_stopping_rounds=30
    )
    if model_lgb_path is None:
        model_lgb_path = MODEL_LGB_PATH
    dump(model_lgb, open(model_lgb_path, 'wb'))

    return model_lgb