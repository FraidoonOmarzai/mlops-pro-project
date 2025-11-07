import os
from box.exceptions import BoxValueError
import yaml
from src.logger import logger
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
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
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
