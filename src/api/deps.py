from functools import lru_cache

from src.inference import BUNDLE_PATH, AttritionPredictor


@lru_cache(maxsize=1)
def get_predictor():
    return AttritionPredictor()


def model_is_ready():
    return BUNDLE_PATH.is_file()
