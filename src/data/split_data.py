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
from sklearn.feature_extraction.text import TfidfVectorizer


le = LabelEncoder()
stop_words = set(stopwords.words('english'))

import string
from nltk.stem import PorterStemmer

string.punctuation

stemmer = PorterStemmer()

def data_processor(data_path):
    news_train = pd.read_csv(data_path)
    news_train['Category_id'] = le.fit_transform(news_train['Category'])

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=stop_words)
    features = tfidf.fit_transform(news_train['Text']).toarray()
    labels = news_train.Category_id

    transformed_data = pd.DataFrame(features)
    transformed_data['labels'] = labels
    return transformed_data


def word_processor(obj):
  txt = "".join([c for c in obj if c not in string.punctuation])
  words = txt.split()
  new_words =[]

  for word in words:
    if word not in set(stopwords.words('english')): 
      new_words.append(stemmer.stem(word))
  return " ".join(new_words)
    

def split_data(df,split_ratio, random_state):
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    return train, test
 

def transform_data(config_path):
    config=read_yaml(config_path)
    
    raw_data_path = config['raw_data_config']['raw_data_csv']
    process_config = config['processed_data_config']
    split_ratio = process_config['train_test_split_ratio']
    random_state = process_config['random_state']
    train_data_path = process_config['train_data_csv']
    test_data_path = process_config['test_data_csv']

    processed_data = data_preprocessor(data_path= raw_data_path)

    train, test = split_data(processed_data, split_ratio, random_state)

    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    transform_data(config_path=parsed_args.config)