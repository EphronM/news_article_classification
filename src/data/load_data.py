import yaml
import argparse
import pandas as pd
from src.config import read_yaml



def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def load_data(data_path):
    df = pd.read_csv(data_path, encoding='utf-8')
    return df

def load_raw_data(config_path):
    config=read_yaml(config_path)

    external_data_path=config["external_data_source"]["cassendra_db"]
    raw_data_path=config["raw_data_config"]["raw_data_csv"]
    
    df = load_data(external_data_path)
    df.to_csv(raw_data_path,index=False)





if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_raw_data(config_path=parsed_args.config)