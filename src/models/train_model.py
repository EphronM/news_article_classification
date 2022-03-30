import json
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.metrics import f1_score,recall_score,accuracy_score,precision_score,confusion_matrix,classification_report
from src.config import read_yaml
from sklearn import svm



def accuracymeasures(y_test, predictions, avg_method):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)


    print("Accuracy Measures")
    print("---------------------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy,precision,recall,f1score


def get_feat_and_target(df,target):
    x=df.drop(target,axis=1)
    y=df[[target]]
    return x,y 


def train_and_evaluate(config_path):
    config = read_yaml(config_path)
    process_config = config['processed_data_config']
    train_data_path = process_config["train_data_csv"]
    test_data_path = process_config["test_data_csv"]
    target = process_config["target"]

    kernel = config["svm_model"]["kernel"]
    gamma = config["svm_model"]["gamma"]
    C = config["svm_model"]["C"]
    

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_x,train_y = get_feat_and_target(train,target)
    test_x,test_y = get_feat_and_target(test,target)


    ############################____MLflow___################################################

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        model = svm.SVC(kernel=kernel, gamma=gamma, C=C)
        model.fit(train_x.values, train_y.values)
        y_pred = model.predict(test_x.values)
        accuracy,precision,recall,f1score = accuracymeasures(test_y.values,y_pred,'weighted')

        mlflow.log_param("kernel",kernel)
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("C", C)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)
       
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(model, "model")

    



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)


#repro