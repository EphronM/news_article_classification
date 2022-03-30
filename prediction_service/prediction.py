import os
import json
import joblib
import numpy as np
from src.config import read_yaml

params_path = 'params.yaml'


config  = read_yaml(params_path)
model_dir_path = config['model_webapp_dir']
vectorizer_path = config['vectorizer_webapp_dir']
mapping_path = config['mapping_path']



def news_prediction(data):
    model = joblib.load(model_dir_path)
    vectorizer = joblib.load(vectorizer_path)
    transformed_data = vectorizer.transform(data).toarray()

    pred = model.predict(transformed_data)
    category = id_to_category(pred)
    return category


def api_response(dict_request):
  try:
    if dict_request:
      data = np.array([list(dict_request.values())])
      pred = news_prediction(data[0])
      if len(pred) == 1:
        response = {'response': pred[0]}
        return response
      elif len(pred) > 1:
        response = {'response': pred}
        return response
  except:
    pass

def id_to_category(pred_id):
    pred_category = []
    id_map = get_maping()
    for id in pred_id:
        pred_category.append(id_map[str(id)])
    return pred_category

def get_maping(json_path = mapping_path):
    with open(json_path) as json_file:
        map = json.load(json_file)
    return map