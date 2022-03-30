from unicodedata import category
import yaml
import os
import json
import joblib
import numpy as np
from src.config import read_yaml

params_path = 'params.yaml'

config  = read_yaml(params_path)
model_dir_path = config['model_webapp_dir']
vectorizer_path = config['model']['vectorizer_dir']
#id_mapping = json.load('mapping.json')



def prediction(data):
  model = joblib.load(model_dir_path)
  vectorizer = joblib.load(vectorizer_path)
  print('inside prediction')

  if vectorizer:
    print('vectorizer exist')
  
  try:
    print('vectorizer started')
    transformed_data = vectorizer.transform([data])
    print(transformed_data.shape)
  except:
    return "Unexpected result"

  prediction = model.predict(transformed_data)
  print(prediction)
  category = id_to_category(prediction)
  return category


def api_response(dict_request):
  try:
    if dict_request:
      data = np.array([list(dict_request.values())])
      print('stage 01')
      response = prediction(data[0])
      response = {'response': response}
      return response
  except:
    print("api responsed with exception")
    pass

def id_to_category(pred_id, pred_category = []):
  id_mapping = get_maping()
  for id in pred_id:
    pred_category.append(id_mapping[id])
  return pred_category

def get_maping(json_path = 'mapping.json'):
    with open(json_path) as json_file:
        map = json.load(json_file)
    return map