import os
import logging
import boto3
import lightgbm as lgb
import joblib


logger = logging.getLogger()

__all__ = ["store", "load"]


def store(model_lgb, filename: str, path: str = "default"):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename + ".joblib")

    logger.info(f"Dumpung model into {filepath}")
    joblib.dump(model_lgb, filepath)

    logging.info("Saving model in Model registry in Yandex Object Storage...")

    session = boto3.session.Session()
    s3 = session.client(
        service_name="s3", endpoint_url="https://storage.yandexcloud.net"
    )
    s3.upload_file(
        filepath,
        "recsys-retail-model-registry",
        "current_model/" + filename + ".joblib",
    )


def load(filename: str, path: str = "default"):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename + ".joblib")

    logger.info(f"Loading model from {filepath}")

    return joblib.load(filepath)


def models_path() -> str:
    script_path = os.path.abspath(__file__)
    script_dir_path = os.path.dirname(script_path)
    models_folder = os.path.join(script_dir_path, "..", "..", "..", "models")
    return models_folder
