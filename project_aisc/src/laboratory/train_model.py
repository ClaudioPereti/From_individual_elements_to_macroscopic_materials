import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import build_models
import tensorflow as tf
import argparse
import yaml
from yaml import Loader
import pathlib
import sys
import mlflow
import numpy as np
import json


def train_parser():
    with open(str(pathlib.Path(
            __file__).absolute().parent.parent.parent) + '/config/available_model_config.yaml') as file:
        model_config = yaml.load(file, Loader)

    my_parser = argparse.ArgumentParser(prog='train model',
                                        description="train model on superconductivity data",
                                        usage='%(prog)s [options]',
                                        )
    my_parser.add_argument('-model',
                           action='store',
                           nargs='+',
                           metavar='MODEL',
                           help="""Select the model. Possible choice:
                                  - """ + '\n-'.join(model_config.keys())
                           )

    my_parser.add_argument('-config',
                           action='store',
                           nargs=1,
                           metavar='CONFIG',
                           help="""Use a custom config for the ML model.
                                   The model need to be specified"""
                           )

    # Parse the args
    args = my_parser.parse_args()

    return args


def main():
    # Path to get inside project_aisc folder
    home_path = str(pathlib.Path(__file__).absolute().parent.parent.parent)

    len_argv = len(sys.argv)
    model_config_path = None

    # Check if any argument is passed from cli
    if len_argv > 1:
        args = train_parser()
        # If ars passed don't contain model we set a default one (regressor)
        if args.model is not None:
            model_name = ' '.join(args.model)
        else:
            model_name = 'regressor'
        model_config_path = args.config
    else:
        model_name = 'regressor'

    # If a custom model is passed through the config file we load it with yaml
    if model_config_path is not None:
        model_config_path = model_config_path[0]
    else:
        model_config_path = home_path + '/config/model_config.yaml'

    with open(model_config_path) as file:
        model_config = yaml.load(file, Loader)

    mlflow.set_tracking_uri(home_path + "/data/experiments/mlruns")
    mlflow.set_experiment(experiment_name=model_name)

    X = list(np.load(home_path + '/data/processed/train/X_train.npy'))
    Y = np.load(home_path + '/data/processed/train/Y_train.npy')
    X_val = list(np.load(home_path + '/data/processed/val/X_val.npy'))
    Y_val = np.load(home_path + '/data/processed/val/Y_val.npy')

    with mlflow.start_run() as run:

        # Define model and train it
        model = build_models.get_model(model_name=model_name, model_config=model_config)
        callbacks = [tf.keras.callbacks.EarlyStopping(**model_config['train setup']['early stopping setup'])]

        # Logs metrics, params, model
        mlflow.tensorflow.autolog(log_models=False)
        model.fit(X, Y, validation_data=(X_val, Y_val), callbacks=callbacks, **model_config['train setup']['fit setup'])
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
    artifact_path = home_path + "/data/experiments/mlruns/" + experiment_id + "/" + run_id + "/artifacts/model"

    mlflow.keras.save_model(
        keras_model=model,
        path=artifact_path,
    )

    with open(home_path + '/../active_experiments.json', 'w') as outfile:
        json.dump({'run id': run_id, 'experiment id': experiment_id}, outfile)
    print({'run id': run.info.run_id, 'experiment id': run.info.experiment_id})


if __name__ == '__main__':
    main()
