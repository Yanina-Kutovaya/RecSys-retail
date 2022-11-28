import os
import logging
from catboost import CatBoostClassifier

__all__ = ["store", "load"]

logger = logging.getLogger()


def store(model_cb, filename: str, path: str = "default"):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename)

    logger.info(f"Dumpung model into {filepath}")
    model_cb.save_model(filepath)


def load(filename: str, path: str = "default"):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename)

    logger.info(f"Loading model from {filepath}")
    model_cb = CatBoostClassifier()

    return model_cb.load_model(filepath)


def models_path() -> str:
    script_path = os.path.abspath(__file__)
    script_dir_path = os.path.dirname(script_path)
    models_folder = os.path.join(script_dir_path, "..", "..", "..", "models")
    return models_folder
