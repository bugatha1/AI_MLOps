import yaml
import os
import json
import logging

def read_config(path_to_file: str) -> dict:
    with open(path_to_file) as yaml_file:
        content = yaml.safe_load(yaml_file)  
    logging.info("yaml file {path_to_file} is loaded successfully")      
    return content

def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"directory is create at {dir_path}")

def save_local_df(data, data_path, index_state=False):
    data.to_csv(data_path, index=index_state)
    logging.info(f"data is save at {data_path}")

def save_reprots(report: dict, report_path: str):
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    logging.info(f"reports are saved at {report_path}")