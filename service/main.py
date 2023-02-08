"""FastAPI RecSys-retail model inference"""

import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

import logging
import pandas as pd
import numpy as np
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from starlette_exporter import PrometheusMiddleware, handle_metrics
from prometheus_client import Counter
from pydantic import BaseModel

from src.recsys_retail.models.serialize import load
from src.recsys_retail.models.inference_tools import preprocess
from src.recsys_retail.metrics import get_recommendations
from src.recsys_retail.models.save_artifacts import save_to_YC_s3

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

MODEL = os.getenv("MODEL", default="baseline_v1")

RECOMMENDATIONS_COUNTER = Counter("recommendations", "Number of recommendations made")
NEW_CLIENTS_COUNTER = Counter("new_clients", "Number of new clients")
N_RECOMMENDATIONS_IN_FILE = 100
MODEL_OUTPUT_S3_BUCKET = "recsys-retail-model-output"


class Model:
    classifier = None


class User(BaseModel):
    user_id: int


class Users(BaseModel):
    user_ids: list


@app.on_event("startup")
def load_model():
    Model.classifier = load(MODEL)


@app.get("/")
def read_healthcheck():
    return {"status": "Green", "version": "0.1.0"}


@app.post("/predict")
def predict(user_id: int, user: User):
    if Model.classifier is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    try:
        id_ = jsonable_encoder(user)["user_id"]
        df, new_user = preprocess(id_)
        if new_user:
            NEW_CLIENTS_COUNTER.inc()
            logging.info(f" The new user: {new_user}")

        predictions = Model.classifier.predict(df)
        results = get_recommendations(df, predictions)
        recs = results["recommendations"][0].tolist()
        recs_dict = {id_: recs}
        RECOMMENDATIONS_COUNTER.inc()
        logging.info(f"User {id_}: {recs}")

        n_recs = 0
        recommendations = []
        ext = 1

        if n_recs < N_RECOMMENDATIONS_IN_FILE:
            recommendations.append([id_, recs])
            n_recs += 1
        else:
            logging.info(
                f"Saving the last {N_RECOMMENDATIONS_IN_FILE} recommendations ..."
            )
            save_to_YC_s3(
                MODEL_OUTPUT_S3_BUCKET,
                file_name=f"recommendations_{ext}",  
                put_object=str(recommendations),
            )
            ext += 1
            recommendations = [[id_, recs]]
            n_recs = 1

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"user_id": id_}, recs_dict


@app.post("/predict_user_list")
def predict_user_list(batch_id: int, users: Users):
    if Model.classifier is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    try:
        ids_ = jsonable_encoder(users)["user_ids"]
        df, new_users = preprocess(ids_, user_list=True)
        if new_users:
            new_users_number = len(new_users)
            NEW_CLIENTS_COUNTER.inc(new_users_number)
            logging.info(f" {new_users_number} new users: {new_users}")

        predictions = Model.classifier.predict(df)
        results = get_recommendations(df, predictions).set_index("user_id")
        recs_ = results.loc[:, "recommendations"]
        recs_dict = {}

        n_recs = 0
        recommendations = []
        ext = 1

        for id in ids_:
            recs = recs_[id].tolist()
            recs_dict[id] = recs
            logging.info(f"User {id}: {recs}")

            if n_recs < N_RECOMMENDATIONS_IN_FILE:
                recommendations.append([id, recs])
                n_recs += 1
            else:
                logging.info(
                    f"Saving the last {N_RECOMMENDATIONS_IN_FILE} recommendations ..."
                )
                save_to_YC_s3(
                    MODEL_OUTPUT_S3_BUCKET,
                    file_name=f"batch_recommendations_{ext}", 
                    put_object=str(recommendations),
                )
                ext += 1
                recommendations = [[id, recs]]
                n_recs = 1

        RECOMMENDATIONS_COUNTER.inc(len(ids_))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"user_ids": ids_}, recs_dict
