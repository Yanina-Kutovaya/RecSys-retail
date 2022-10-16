import joblib
from typing import Optional
from src.recsys_retail.features.recommenders import MainRecommender

logger = logging.getLogger(__name__)

__all__ = ['train_save_recommender']

args = {
    'n_factors_ALS': 50,
    'regularization_ALS': 0.001, 
    'iterations_ALS': 20, 
    'num_threads_ALS': 4
}   
    
    
def train_save_recommender(data_train_lvl_1: pd.DataFrame, args):

    logging.info('Training recommender...')           
    recommender = MainRecommender(data_train_lvl_1, args)

    logging.info('Saving recommender...')            
    if recommender_path is None:
        recommender_path = RECOMMENDER_PATH
    joblib.dump(recommender, recommender_path, 3)

    return recommender