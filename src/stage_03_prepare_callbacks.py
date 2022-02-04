from src.utils.all_utils import rea read_config, create_directory
from src.utils.models import get_VGG_16_model, prepare_model
from src.utils.callbacks import create_and_save_tensorboard_callback, create_and_save_checkpoint_callback
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
    config = read_config(config_path)
    params = read_config(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    tensorboard_log_dir = os.path.join(artifacts_dir, artifacts["TENSORBOARD_ROOT_LOG_DIR"])
    checkpoint_dir = os.path.join(artifacts_dir, artifacts["CHECKPOINT_DIR"])
    callbacks_dir = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    
    create_directory(
        tensorboard_log_dir,
        checkpoint_dir,
        callbacks_dir
    )

    create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir)
    create_and_save_checkpoint_callback(callbacks_dir, checkpoint_dir)




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