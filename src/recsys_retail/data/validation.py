import pandas as pd
from sklearn.model_selection import train_test_split as sklean_train_test_split
from typing import Tuple

__all__ = ["train_test_split"]


def train_test_split(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 16
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    X_train, X_valid, y_train, y_valid = sklean_train_test_split(
        df.drop('target', axis=1).fillna(0), 
        df[['target']], 
        test_size=test_size,
        random_state=random_state, 
        stratify=dataset[['target']]
    )

    return X_train, X_valid, y_train, y_valid