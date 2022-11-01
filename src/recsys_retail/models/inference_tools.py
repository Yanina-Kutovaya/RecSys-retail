import pandas as pd
from src.recsys_retail.models.load_artifacts import load_inference_artifacts
from src.recsys_retail.features.candidates_lvl_2 import get_candidates
from src.recsys_retail.features.targets import get_targets_lvl_2
from src.recsys_retail.features.new_item_user_features import get_embeddings

N_ITEMS = 100


def preprocess(user_ids, user_list=False) -> pd.DataFrame:
    """
    Preprocesses user_id  for inference 
    
    """
    (data_valid, recommender, user_features_transformed, 
    item_features_transformed, user_item_features) = load_inference_artifacts()

    if not user_list:
        user_ids = [user_ids]
    df = pd.DataFrame(user_ids, index=range(len(user_ids)), columns=['user_id'])

    users_inference = get_candidates(
        recommender, prefiltered_data, df, n_items=N_ITEMS
    )
    train_dataset_lvl_2 = get_targets_lvl_2(
        users_inference, 
        data_valid,
        item_features_transformed, 
        user_features_transformed,     
        user_item_features, 
        n_items=N_ITEMS
    )   
    train_dataset_lvl_2 = get_embeddings(
    recommender, train_dataset_lvl_2
    )

    return train_dataset_lvl_2.drop('target', axis=1).fillna(0)

        
