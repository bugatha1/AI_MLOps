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



def prepare_base_model(config_path, params_path):
    config = read_config(config_path)
    params = read_config(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    base_model_dir = artifacts["BASE_MODEL_DIR"]
    base_model_name = artifacts["BASE_MODEL_NAME"]

    base_model_dir_path = os.path.join(artifacts_dir, base_model_dir)
    create_directory([base_model_dir_path])

    base_model_path = os.path.join(base_model_dir_path, base_model_name)

    model = get_VGG_16_model(input_shape=params["IMAGE_SIZE"], model_path=base_model_path)
    
    full_model = prepare_model(
        model,
        CLASSES = params["CLASSES"],
        freeze_all= True,
        freeze_till = None,
        learning_rate = params["LEARNING_RATE"]
    )

    update_base_model_path = os.path.join(
        base_model_dir_path,
        artifacts["UPDATED_BASE_MODEL_NAME"]
    )

    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

    logging.info(f" full model summary : \n{_log_model_summary(full_model)}")

    full_model.save(update_base_model_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info("stage_02 task is started")
        prepare_base_model(parsed_args.config, parsed_args.params)
        logging.info("stage_02 task is completed. Prepared base model")
    except Exception as e:
        logging.exception(e)
        raise e