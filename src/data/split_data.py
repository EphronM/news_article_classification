import os
import argparse
import pandas as pd
from load_data import read_params
from sklearn.model_selection import train_test_split

def split_data(df,train_data_path,test_data_path,split_ratio,random_state):
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")    

def split_and_saved_data(config_path):

    config = read_params(config_path)
    process_config = config['processed_data_config']
    data_path = process_config["processed_data_csv"]
    test_data_path = process_config["test_data_csv"] 
    train_data_path = process_config["train_data_csv"]
    split_ratio = process_config["train_test_split_ratio"]
    random_state = process_config["random_state"]

    new_data_df = pd.read_csv(data_path)
    split_data(new_data_df, train_data_path, test_data_path, split_ratio, random_state)
    



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)