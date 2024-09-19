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
    # Check if cache folder exists.
    cache_folder_path = os.path.join(os.getcwd(), "cache")
    # Attempt to create the folder if it doesn't exist.
    if not os.path.isdir(cache_folder_path):
        try:
            os.mkdir(cache_folder_path)
            logger.info(f"Created cache folder at location '{cache_folder_path}'.")
        except Exception as e:
            # If the cache folder could not be created, print a warning and exit (cache will not be used).
            logger.warning(f"Cache folder '{cache_folder_path}' does not exist and could not be created. Reason: '{e}'.")
            return
    
    logger.info(f"Saving cache to location '{path}'")
    try:
        with open(path, "wb") as file:
            return pickle.dump(data, file)
    except Exception as e:
        logger.warning(f"Failed to save cache to '{path}': '{e}'")
