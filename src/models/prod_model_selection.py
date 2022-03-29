import joblib
import mlflow
import argparse
from pprint import pprint
from src.config import read_yaml
from mlflow.tracking import MlflowClient
import pandas as pd

def log_production_model(config_path):
    config = read_yaml(config_path)
    mlflow_config = config["mlflow_config"] 
    model_name = mlflow_config["registered_model_name"]
    model_dir = config['model']["model_dir"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    params_dir = config['model']["params_dir"]
    score_dir = config['model']["score_dir"]

    mlflow.set_tracking_uri(remote_server_uri)
    runs = mlflow.search_runs(experiment_ids='1')
    max_accuracy = max(runs["metrics.accuracy"])
    max_accuracy_run_id = list(runs[runs["metrics.accuracy"] == max_accuracy]["run_id"])[0]
    print("---------------------------------","\n")
    pref_run = runs[runs["run_id"]==max_accuracy_run_id]

    params = {
        'kernel' : pref_run['params.kernel'],
        'gamma' : pref_run['params.gamma'],
        'C' : pref_run['params.C']
    }
    score = {
        'Accuracy' : pref_run['metrics.accuracy'],
        'F1_score' : pref_run['metrics.f1_score'],
        'Recall' : pref_run['metrics.recall'],
        'Precision' : pref_run['metrics.precision'],
    }

    pd.DataFrame(params).to_json(params_dir)
    pd.DataFrame(score).to_json(score_dir)

    print(params)
    print(score)
    
    print("---------------------------------","\n")
    
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)

        if mv["run_id"] == max_accuracy_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )        

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    joblib.dump(loaded_model, model_dir)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)