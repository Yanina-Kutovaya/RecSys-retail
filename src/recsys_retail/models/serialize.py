import os
import logging
import boto3
import lightgbm as lgb

__all__ = ["store", "load"]

logger = logging.getLogger()


def store(model_lgb, filename: str, path: str = "default"):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename + ".txt")

    logger.info(f"Dumpung model into {filepath}")
    model_lgb.save_model(filepath)

    session = boto3.session.Session()
    s3 = session.client(
        service_name="s3", endpoint_url="https://storage.yandexcloud.net"
    )
    s3.upload_file(
        filepath, "recsys-retail-model-registry", "current_model/" + filename
    )


def load(filename: str, path: str = "default"):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename + ".txt")

    logger.info(f"Loading model from {filepath}")    

    return lgb.Booster(model_file=filepath)


def models_path() -> str:
    script_path = os.path.abspath(__file__)
    script_dir_path = os.path.dirname(script_path)
    models_folder = os.path.join(script_dir_path, "..", "..", "..", "models")
    return models_folder
