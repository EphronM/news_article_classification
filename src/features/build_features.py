import argparse
import pandas as pd 
from src.config import read_yaml
import joblib
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


le = LabelEncoder()
stop_words = set(stopwords.words('english'))
string.punctuation
stemmer = PorterStemmer()



def data_processor(data_path):
  news_train = pd.read_csv(data_path)
  news_train['Category_id'] = le.fit_transform(news_train['Category'])
  news_train['Text'] = news_train['Text'].apply(word_processor)

  tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=stop_words)
  features = tfidf.fit_transform(news_train['Text']).toarray()
  labels = news_train.Category_id

  transformed_data = pd.DataFrame(features)
  transformed_data['labels'] = labels

  return transformed_data, tfidf


def word_processor(obj):
  txt = "".join([c for c in obj if c not in string.punctuation])
  words = txt.split()
  new_words =[]

  for word in words:
    if word not in set(stopwords.words('english')): 
      new_words.append(stemmer.stem(word))
  return " ".join(new_words)
    

def transform_data(config_path):
  config=read_yaml(config_path)
    
  raw_data_path = config['raw_data_config']['raw_data_csv']
  process_config = config['processed_data_config']
  new_data_path = process_config['processed_data_csv']
  vectorizer_path = config['model']['vectorizer_dir']
    
  processed_data, vectorizer = data_processor(data_path= raw_data_path)

  processed_data.to_csv(new_data_path, sep=",", index=False, encoding="utf-8")
  joblib.dump(vectorizer, vectorizer_path)




if __name__ == "__main__":
  args = argparse.ArgumentParser()
  args.add_argument("--config", default="params.yaml")
  parsed_args = args.parse_args()
  transform_data(config_path=parsed_args.config)


#repro