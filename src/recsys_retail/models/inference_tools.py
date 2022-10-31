import pandas as pd
from src.recsys_retail.models.load_artifacts import load_inference_artifacts
from src.recsys_retail.features.candidates_lvl_2 import get_candidates
from src.recsys_retail.features.targets import get_targets_lvl_2

N_ITEMS = 100


def preprocess(user_id) -> pd.DataFrame:
    """
    Preprocesses user_id  for inference 
    
    """
    (prefiltered_data, recommender, user_features_transformed, 
    item_features_transformed, user_item_features) = load_inference_artifacts()

    if not type(user_id) == list:
        user_id = [user_id]
    df = pd.DataFrame(user_id, index=range(len(user_id)), columns=['user_id'])
        
    users_inference = get_candidates(
        recommender, prefiltered_data, df, n_items=N_ITEMS
    )
    train_dataset_lvl_2 = get_targets_lvl_2(
        users_inference, 
        prefiltered_data,
        item_features_transformed, 
        user_features_transformed,     
        user_item_features, 
        n_items=N_ITEMS
    )

    return train_dataset_lvl_2.drop('target', axis=1).fillna(0)

        
