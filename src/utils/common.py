import os
import sys
from box.exceptions import BoxValueError
import yaml
from src.logger import logger
from src.exception import CustomException
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import json
import pickle


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def save_json(path: str, data: dict):
    """
    Save a dictionary as a JSON file.
    """
    # ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def save_object(file_path: str, obj):
    """
    Save any Python object to a file using pickle.

    Args:
        file_path (str): Path to the file where the object should be saved.
        obj: Python object to pickle.
    """
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(file_path)
    if dir_path != "":
        os.makedirs(dir_path, exist_ok=True)

    # Save object
    with open(file_path, 'wb') as file_obj:
        pickle.dump(obj, file_obj)

@ensure_annotations
def load_object(file_path: str):
    """
    Load a Python object from a pickle file.

    Args:
        file_path (str): Path to the .pkl file

    Returns:
        Any: The loaded Python object
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)