from unicodedata import category
import yaml
import os
import json
import joblib
import numpy as np
from src.config import read_yaml

params_path = 'params.yaml'

config  = read_yaml(params_path)
model_dir_path = config['webapp_model_dir']
vectorizer_path = config['model']['vectorizer_dir']
id_mapping = json.load('mapping.json')

def prediction(data):
    model = joblib.load(model_dir_path)
    vectorizer = joblib.load(vectorizer_path)

    transformed_data = vectorizer.transform([data]).toarray()
    prediction = model.predict(transformed_data)
    category = id_to_category(prediction)
    return category


def api_response(dict_request):
    try:
        if dict_request:
            data = np.array([list(dict_request.values())])
            response = prediction(data)
            response = {'response': response}
            return response
    except:
        pass

def id_to_category(pred_id, pred_category = []):
  for id in pred_id:
    pred_category.append(id_mapping[id])
  return pred_category
