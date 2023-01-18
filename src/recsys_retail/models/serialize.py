import os
import logging
import lightgbm as lgb
import joblib

from src.recsys_retail.models.save_artifacts import save_to_YC_s3


logger = logging.getLogger()

__all__ = ["store", "load"]

MODEL_REGISTRY = "recsys-retail-model-registry"


def store(model_lgb, filename: str, path: str = "default"):
    if path == "default":
        path = models_path()

    filepath = os.path.join(path, filename + ".joblib")

    logger.info(f"Dumpung model into {filepath}")
    joblib.dump(model_lgb, filepath)

    logging.info("Saving model in Model registry in Yandex Object Storage...")

    save_to_YC_s3(
        MODEL_REGISTRY, path, file_name=filename + ".joblib", s3_path="current_model/"
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
