import os
import pickle

from loguru import logger

def cache_exists(path: str) -> bool:
    return os.path.isfile(path)

def load_cache(path: str):
    """
    Returns None if cache failed to load.
    """
    logger.info(f"Loading cache at location '{path}'")
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        logger.warning(f"Failed to load cache from '{path}': '{e}'")
        return None

def save_cache(data, path: str):
    logger.info(f"Saving cache to location '{path}'")
    try:
        with open(path, "wb") as file:
            return pickle.dump(data, file)
    except Exception as e:
        logger.warning(f"Failed to save cache to '{path}': '{e}'")
