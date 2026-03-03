import os 
import yaml
from src.wine_quality_project import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads yaml file and returns

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
            logger.info(f"yaml file: {path_to_yaml} loaded successfuly")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError('yaml file is empty')
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    create list of directories

    Args:
        path_to_directories (str): list of path of directories
        ignore_log (bool, optional): ignore if multiple directories is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f'created directories at: {path}')


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    save json data

    Args:
        path: path to json file
        data: datat to be saved as json file
    """
    with open(path, "w") as file:
        json.dump(data, file, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    load json files data

    Args:
        path: path to json file
    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as file:
        content = json.load(file)

    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    save binary file

    Args:
        data (Any): data to be saved as a binary file
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    load binary data
    
    Args:
        path (Path): path to binary data file
    Returns:
        object stored in file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data