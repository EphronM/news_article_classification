from urllib import response
from flask import Flask, render_template, request, jsonify
import os
import yaml
import joblib
import numpy as np
from src.config import read_yaml
from prediction_service import prediction



#webapp_root = "webapp"


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if request.form:
                data_req = dict(request.form)
                response = prediction.form_response(data_req)
                return render_template('index.html', response = response)
            elif request.json:
                print('data fetched')
                response = prediction.api_response(request.json)
  
                return jsonify(response)
        except:
            pass

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

