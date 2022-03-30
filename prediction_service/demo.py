
import yaml
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
#id_mapping = json.load('mapping.json')



def news_prediction(data):
    print('inside prediction')
    model = joblib.load(model_dir_path)
    vectorizer = joblib.load(vectorizer_path)
    print('artifacts loaded')

    if vectorizer:
        print('vectorizer exist')
  

    print('vectorizer started')
    #print(vectorizer.transform(data).toarray())
    transformed_data = vectorizer.transform(data).toarray()
    print(transformed_data.shape)
    
    if model:
        print(model)

    print('--------------------------------')
    pred = model.predict(transformed_data)
    print(pred)
    print('--------------------------------')

    category = id_to_category(pred)
    return category


def api_response(dict_request):
  try:
    if dict_request:
      data = np.array([list(dict_request.values())])
      print('stage 01')
      response = news_prediction(data[0])
      response = {'response': response}
      return response
  except:
    print("api responsed with exception")
    pass

def id_to_category(pred_id, pred_category = []):
    id_map = get_maping()
    for id in pred_id:
        pred_category.append(id_map[str(id)])
    print(pred_category)
    return pred_category

def get_maping(json_path = mapping_path):
    with open(json_path) as json_file:
        map = json.load(json_file)
    return map

if __name__=="__main__":

    request = {
                "text": "worldcom ex-boss launches defence lawyers defending former worldcom chief bernie ebbers against a battery of fraud charges have called a company whistleblower as their first witness.  cynthia cooper  worldcom s ex-head of internal accounting  alerted directors to irregular accounting practices at the us telecoms giant in 2002. her warnings led to the collapse of the firm following the discovery of an $11bn (拢5.7bn) accounting fraud. mr ebbers has pleaded not guilty to charges of fraud and conspiracy.  prosecution lawyers have argued that mr ebbers orchestrated a series of accounting tricks at worldcom  ordering employees to hide expenses and inflate revenues to meet wall street earnings estimates. but ms cooper  who now runs her own consulting business  told a jury in new york on wednesday that external auditors arthur andersen had approved worldcom s accounting in early 2001 and 2002. she said andersen had given a  green light  to the procedures and practices used by worldcom. mr ebber s lawyers have said he was unaware of the fraud  arguing that auditors did not alert him to any problems.  ms cooper also said that during shareholder meetings mr ebbers often passed over technical questions to the company s finance chief  giving only  brief  answers himself. the prosecution s star witness  former worldcom financial chief scott sullivan  has said that mr ebbers ordered accounting adjustments at the firm  telling him to  hit our books . however  ms cooper said mr sullivan had not mentioned  anything uncomfortable  about worldcom s accounting during a 2001 audit committee meeting. mr ebbers could face a jail sentence of 85 years if convicted of all the charges he is facing. worldcom emerged from bankruptcy protection in 2004  and is now known as mci. last week  mci agreed to a buyout by verizon communications in a deal valued at $6.75bn."

                }


    api_response(request)