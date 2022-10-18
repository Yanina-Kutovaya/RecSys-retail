import pandas as pd
import joblib
from typing import Optional


PATH = 'models/'
RECOMMENDER_PATH = PATH + 'recommender_v1' 

N_ITEMS = 100

def preprocess(
    test: pd.DataFrame,
    
    n_items: Optional[int] = None
    ) -> pd.DataFrame:

    """
    Preprocesses transactions data for inference with binary classifier
    """ 

    (
        recommender, 
        data_train, data_valid, 
        item_features_transformed, 
        user_features_transformed,    
        user_item_features_final
    ) = load_artefacts()

    if n_items is None:
        n_items = N_ITEMS

    users_inference = get_candidates(
        recommender, data_train, test, n_items
    )
    test_dataset_inference = get_targets_lvl_2(
        users_inference, 
        data_valid,
        item_features_transformed, 
        user_features_transformed,    
        user_item_features_final,     
        n_items
    )

    return test_dataset_inference.drop('target', axis=1).fillna(0)


def load_artefacts(
    recommender_path: Optional[str] = None,
    data_train_path: Optional[str] = None,
    data_valid_path: Optional[str] = None,
    item_features_transformed_path: Optional[str] = None,
    user_features_transformed_path: Optional[str] = None,
    user_item_features_final_path: Optional[str] = None,
):

    if recommender_path is None:
        recommender_path = RECOMMENDER_PATH
    recommender = joblib.load(recommender_path)

    return (
        recommender, 
        data_train, data_valid, 
        item_features_transformed, 
        user_features_transformed,    
        user_item_features_final
    ) 