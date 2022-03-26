import yaml
import argparse
import numpy as np 
import pandas as pd 
from src.config import read_yaml
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def data_preprocessor(data_path):
    pass

def split_data(df, train_data_path, test_data_path, split_ratio, random_state):
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    return train, test

    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8") 

def transform_data(config_path):
    config=read_yaml(config_path)
    
    raw_data_path = config['raw_data_config']['raw_data_csv']
    process_config = config['processed_data_config']
    split_ratio = process_config['train_test_split_ratio']
    random_state = process_config['random_state']
    target = process_config['target']
    train_data_path = process_config['train_data_csv']
    test_data_path = process_config['test_data_csv']

    processed_data = data_preprocessor(data_path= raw_data_path)
    train, test = split_data(processed_data,train_data_path,test_data_path,split_ratio,random_state)








if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    transform_data(config_path=parsed_args.config)