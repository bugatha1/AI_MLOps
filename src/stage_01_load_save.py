from src.utils.all_utils import read_config, create_directory
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

def copy_file(source_download_dir, local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    N = len(list_of_files)
    for file in tqdm(list_of_files, total=N, desc=f" copying file from {source_download_dir} to {local_data_dir}",
    colour="green" ):
        src = os.path.join(source_download_dir, file)
        dest = os.path.join(local_data_dir, file)
        shutil.copy(src, dest)

def get_data(config_path):
    config = read_config(config_path)
    
    source_download_dirs = config["source_download_dirs"]
    local_data_dirs = config["local_data_dirs"]

    for source_download_dir, local_data_dir in tqdm(zip(source_download_dirs, local_data_dirs), total=2, desc="iterate folders", 
    colour="red"):
        create_directory([local_data_dir])
        copy_file(source_download_dir, local_data_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info("stage_01 task is started")
        get_data(parsed_args.config)
        logging.info("stage_01 task is completed. All data is saved in local")
    except Exception as e:
        logging.exception(e)
        raise e