import logging
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'..'))

from fastapi.testclient import TestClient

from src.recsys_retail.data.make_dataset import load_data
from src.recsys_retail.models.train import data_preprocessing_pipeline

from main import app, Model 


logger = logging.getLogger(__name__)

client = TestClient(app)


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
    Model.classifier.fit(train_dataset_lvl_2)

    transaction = {
        'user_id': 1340,
        'basket_id': 41652823310,
        'day': 664,
        'item_id': 912987,	
        'quantity': 1,	
        'sales_value': 8.49,	
        'store_id': 446,	
        'retail_disc': 0.0,	
        'trans_time': 52,	
        'week_no': 96
    }
    response = client.post('/predict?user_id=1340', json=transaction)
    assert response.status_code == 200
    
    assert response.json()['user_id'] == 1340

    print(f'recommendations = {response.json()["recommendations"]}')