import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import mlflow
import numpy as np
import pathlib
import tensorflow as tf
import json
import argparse
import sys
import warnings

def evaluate_parser():
    my_parser = argparse.ArgumentParser(prog='evaluate model',
                                        description="evaluate model on superconductivity data",
                                        usage='%(prog)s [options]',
                                        )

    my_parser.add_argument('-problem',
                           action='store',
                           nargs=1,
                           metavar='CONFIG',
                           choices=['regression', 'classification'],
                           help="""Set the type of problem, classification or regression"""
                           )

    my_parser.add_argument('--no-save',
                           action='store',
                           nargs='?',
                           help="Don't save/track results with mlflow. Arguments specify what to not track",
                           choices=['artifacts', 'model','all'],
                           const='all'
                           )

    # Parse the args
    args = my_parser.parse_args()

    return args


def get_metrics(Y_pred, Y_true, problem='classification'):
    if problem == 'regression':
        mae = tf.keras.metrics.MeanAbsoluteError()(Y_pred, Y_true)
        mse = tf.keras.metrics.MeanSquaredError()(Y_pred, Y_true)
        return {'mae': float(mae), 'mse': float(mse), 'rmse': float(tf.math.sqrt(mse))}
    elif problem == 'classification':
        threshold = 0.5
        precision = tf.keras.metrics.Precision(thresholds=threshold)(Y_true, Y_pred)
        recall = tf.keras.metrics.Recall(thresholds=threshold)(Y_true, Y_pred)
        return {'precision': float(precision), 'recall': float(recall)}


def main():
    # Path to get inside project_aisc folder
    home_path = str(pathlib.Path(__file__).absolute().parent.parent.parent)
    with open(home_path + "/../active_experiments.json") as experiments_json:
        experiment_info = json.load(experiments_json)

    run_id = experiment_info["run id"]
    experiment_id = experiment_info["experiment id"]
    model_uri = home_path + "/data/experiments/mlruns/" + experiment_id + "/" + run_id + "/artifacts/model"
    model = mlflow.keras.load_model(model_uri)

    len_argv = len(sys.argv)
    if len_argv > 1:
        args = evaluate_parser()
        if args.problem is not None:
            problem = args.problem[0]
        else:
            problem = 'regression'
            warnings.warn("Problem not specified! Default problem is regression")
        if args.no_save is not None:
            no_save = args.no_save
        else:
            no_save = None
    else:
        problem = 'regression'
        warnings.warn("Problem not specified! Default problem is regression")
        no_save = None

    if problem == "regression":
        base_data_test_name = 'Supercon'
    elif problem == "classification":
        base_data_test_name = 'Supercon + Garbage in'

    print(f"Evaluation model on {base_data_test_name} datase...")
    X_test = list(np.load(home_path + '/data/processed/test/X_test.npy'))
    Y_test = np.load(home_path + '/data/processed/test/Y_test.npy')

    Y_pred = model.predict(X_test)
    metrics = get_metrics(Y_pred=Y_pred, Y_true=Y_test, problem=problem)
    print("Evaluation on Test set:")
    [print(f"{item[0]} : {item[1]}") for item in metrics.items()]

    MlflowClient = mlflow.tracking.client.MlflowClient(tracking_uri=home_path + "/data/experiments/mlruns/")

    [MlflowClient.log_metric(run_id=run_id, key='test_' + metric, value=value) for metric, value in metrics.items()]

    artifact_uri = home_path + "/data/experiments/mlruns/" + experiment_id + "/" + run_id + "/artifacts"
    if no_save is None:
        # Save predictions of the model
        np.save(artifact_uri + "/predictions.npy", tf.reshape(model.predict(X_test), shape=(-1,)).numpy())
    print(f"Evaluation on {base_data_test_name} dataset completed.")

    try:
        print("Evaluation on Hosono dataset...")
        X_hosono = list(np.load(home_path + "/data/processed/hosono.npy"))
        Y_hosono = np.load(home_path + '/data/processed/Y_hosono.npy')

        Y_hosono_pred = model.predict(X_hosono)
        metrics = get_metrics(Y_pred=Y_hosono_pred, Y_true=Y_hosono, problem=problem)
        print("Evaluation on Hosono:")
        [print(f"{item[0]} : {item[1]}") for item in metrics.items()]
        [MlflowClient.log_metric(run_id=run_id, key='hosono_' + metric, value=value) for metric, value in metrics.items()]
        print(f"Evaluation on Hosono dataset completed.")
    except FileNotFoundError:
        print("Hosono data not found. Skipping test on this dataset.")

    try:
        print("Evaluation on IMA dataset...")
        X_ima = list(np.load(home_path + "/data/processed/ima.npy"))
        Y_ima_pred = tf.reshape(model.predict(X_ima), shape=(-1,))
        if no_save is None:
            np.save(artifact_uri + "/ima_predictions.npy", tf.reshape(model.predict(X_test), shape=(-1,)).numpy())

        print(Y_ima_pred)
        print(f"Evaluation on IMA dataset completed.")
    except FileNotFoundError:
        print("IMA data not found. Skipping test on this dataset.")

    print(f"run id: {run_id}")
    if no_save == 'model':
        try:
            shutil.rmtree(home_path + "/data/experiments/mlruns/" + experiment_id + "/" + run_id + "/artifacts/model")
        except:
            print("Error")
    elif no_save == 'all':
        try:
            shutil.rmtree(home_path + "/data/experiments/mlruns/" + experiment_id + "/" + run_id)
        except:
            print("Error")


if __name__ == '__main__':
    main()
