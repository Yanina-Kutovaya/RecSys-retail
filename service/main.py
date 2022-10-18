"""FastAPI Recommender systen retail model inference"""

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'..'))

import pandas as pd
import numpy as np
from typing import Optional

from fastapi import FastAPI, HTTPException
from starlette_exporter import PrometheusMiddleware, handle_metrics
from pydantic import BaseModel

from src.recsys_retail.models.serialize import load
from src.recsys_retail.models.inference_data import preprocess
from src.recsys_retail.metrics import get_results


app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

MODEL = os.getenv("MODEL", default='LightGBM_v1')


class Model:
    classifier = None


class Transaction(BaseModel):
    user_id: int
    basket_id: int
    day: int
    item_id: int	
    quantity: int	
    sales_value: float	
    store_id: int	
    retail_disc: float	
    trans_time: int	
    week_no: int	
    coupon_disc: Optional[float] = None	
    coupon_match_disc: Optional[float] = None 


@app.on_event('startup')
def load_model():
    Model.classifier = load(MODEL)


@app.get('/')
def read_healthcheck():
    return {'status': 'Green', 'version': '0.1.0'}


@app.post('/predict')
def predict(user_id: int, transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    data.fillna(value=np.nan, inplace=True, downcast=False)   

    if Model.classifier is None:
        raise HTTPException(status_code=503, detail='No model loaded')
    try:        
        df = preprocess(data)
        predictions = Model.classifier.predict(df)
        results = get_results(data, df, predictions)
        recommendations = results.loc[results['user_id']==user_id, 'recommendations'].values
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {'user_id': user_id, 'recommendations': recommendations}
