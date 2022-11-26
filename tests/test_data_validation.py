import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

import pytest
import numpy as np
import pandas as pd
from src.recsys_retail.data.validation import train_test_split


@pytest.mark.parametrize("input_size", [100, 101])
def test_data_split(input_size):
    df = pd.DataFrame(np.random.randn(input_size, 2))
    df["target"] = np.random.randint(2, size=input_size)
    X_train, X_test, y_train, y_test = train_test_split(df)
    assert input_size == len(X_train) + len(X_test)
    assert input_size == len(y_train) + len(y_test)


@pytest.mark.parametrize("input_size", [0, 1])
def test_data_split_fail(input_size):
    df = pd.DataFrame(np.random.randn(input_size, 2))
    df["target"] = np.random.randint(2, size=input_size)
    with pytest.raises(ValueError):
        X_train, X_test, y_train, y_test = train_test_split(df)
