from src.utils.all_utils import read_config, create_directory
from src.utils.models import get_VGG_16_model, prepare_model
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm
import logging
import io

logging_str = "[%(asctime)s : %(levelname)s: %(module)s] : %(message)s"
logging_dir = "logs"
os.makedirs(logging_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logging_dir, "runninglog.log"), level=logging.INFO,
format=logging_str, filemode="a")

def prepare_callbacks(config_path, params_path):
    pass


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info("stage_03 task is started")
        prepare_callbacks(parsed_args.config, parsed_args.params)
        logging.info("stage_03 task is completed. Prepared base model")
    except Exception as e:
        logging.exception(e)
        raise e