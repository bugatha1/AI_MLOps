from src.utils.all_utils import read_config, create_directory
from src.utils.models import load_full_model
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm
import logging

logging_str = "[%(asctime)s : %(levelname)s: %(module)s] : %(message)s"
logging_dir = "logs"
os.makedirs(logging_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logging_dir, "runninglog.log"), level=logging.INFO,
format=logging_str, filemode="a")

def train_model(config_path, params_path):
    config = read_config(config_path)
    params = read_config(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    train_model_dir_path = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directory([train_model_dir_path])

    untrained_full_model_path = os.path.join(artifacts_dir, artifacts["BASE_MODEL_DIR"], artifacts["UPDATED_BASE_MODEL_NAME"])
    model = load_full_model(untrained_full_model_path)

    callbacks_dir_path = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks(callbacks_dir_path)

    train_generator, valid_generator = train_valid_generator(
        data_dir = artifacts["DATA_DIR"],
        IMAGE_SIZE = tuple(params["IMAGE_SIZE"][:-1]),
        BATCH_SIZE = params["BATCH_SIZE"],
        do_data_augmentation = params["AUGMENTATION"]
    )

  


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info("stage_04 task is started")
        train_model(parsed_args.config, parsed_args.params)
        logging.info("stage_04 task is completed. trainnig is completed")
    except Exception as e:
        logging.exception(e)
        raise e