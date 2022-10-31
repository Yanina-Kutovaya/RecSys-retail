"""FastAPI RecSys-retail model inference"""

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'..'))

import pandas as pd
import numpy as np
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from starlette_exporter import PrometheusMiddleware, handle_metrics
from pydantic import BaseModel

from src.recsys_retail.models.serialize import load
from src.recsys_retail.models.inference_tools import preprocess
from src.recsys_retail.metrics import get_recommendations


app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

MODEL = os.getenv("MODEL", default='LightGBM_v1')


class Model:
    classifier = None


class User(BaseModel):
    user_id: int


class Users(BaseModel):
    user_ids: list
    
 
@app.on_event('startup')
def load_model():
    Model.classifier = load(MODEL)


@app.get('/')
def read_healthcheck():
    return {'status': 'Green', 'version': '0.1.0'}


@app.post('/predict')
def predict(user_id: int, user: User):
    if Model.classifier is None:
        raise HTTPException(status_code=503, detail='No model loaded')
    try:
        user_id = jsonable_encoder(user)['user_id']       
        df = preprocess(user_id)
        predictions = Model.classifier.predict(df)
        results = get_recommendations(df, predictions)        
        recs = {results['user_id'][0]: results['recommendations'][0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    return {'user_id': user_id, 'recommendations': recs}


@app.post('/predict_user_list')
def predict(user_ids: list, users: Users):
    if Model.classifier is None:
        raise HTTPException(status_code=503, detail='No model loaded')
    try:
        user_ids = jsonable_encoder(users)['user_ids']        
        df = preprocess(user_ids, user_list=True)
        predictions = Model.classifier.predict(df)
        results = get_recommendations(df, predictions)        
        recs = {results['user_id']: results['recommendations'].tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    return {'user_ids': user_ids, 'recommendations': recs}
