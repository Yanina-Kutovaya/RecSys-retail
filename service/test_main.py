import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

import logging
import pandas as pd
from fastapi.testclient import TestClient

from main import app, Model
from src.recsys_retail.models.serialize import load
from src.recsys_retail.data.make_dataset import load_data
from src.recsys_retail.models.train import data_preprocessing_pipeline
from scripts.train_save_model import train_store


logger = logging.getLogger(__name__)

client = TestClient(app)

MODEL = os.getenv("MODEL", default="LightGBM_v1")


def test_healthcheck():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Green"
    logger.info(f'status_code = 200, status = "Green"')


def test_predict():
    data, item_features, user_features = load_data()
    train_dataset_lvl_2 = data_preprocessing_pipeline(
        data, item_features, user_features
    )
    train_store(train_dataset_lvl_2, "LightGBM_v1")
    Model.classifier = load(MODEL)

    user = {"user_id": 1340}
    response = client.post("/predict?user_id=1340", json=user)
    assert response.status_code == 200
    assert response.json()[0]["user_id"] == 1340

    logger.info(
        f"test_predict single user status_code = 200, response: {response.json()}"
    )

    users = {"user_ids": [1340, 1364]}
    response = client.post("/predict_user_list?batch_id=1", json=users)
    assert response.status_code == 200
    assert response.json()[0]["user_ids"] == [1340, 1364]

    logger.info(
        f"test_predict user_list status_code = 200, response: {response.json()}"
    )
