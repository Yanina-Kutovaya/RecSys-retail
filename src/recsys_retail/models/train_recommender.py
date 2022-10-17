import logging
import pandas as pd
import joblib
from typing import Optional
from src.recsys_retail.features.recommenders import MainRecommender

logger = logging.getLogger(__name__)

__all__ = ['train_save_recommender']

PATH = 'models/'
RECOMMENDER_PATH = PATH + 'recommender_v1'   
    
    
def train_save_recommender(
    data_train_lvl_1: pd.DataFrame,
    recommender_path: Optional[str] = None
    ):
    """
    Generates recommender which will be used for selecting of 
    long list of items for each user (the 1st stage).
    On the 2nd stage this long list will be the bases for the further 
    short list selection with the binary classification model. 

    """

    logging.info('Training recommender...')           
    recommender = MainRecommender(
        data_train_lvl_1, 
        n_factors_ALS=50, 
        regularization_ALS=0.001,
        iterations_ALS=15,
        num_threads_ALS=4 
    )

    logging.info('Saving recommender...')            
    if recommender_path is None:
        recommender_path = RECOMMENDER_PATH
    joblib.dump(recommender, recommender_path, 3)

    return recommender