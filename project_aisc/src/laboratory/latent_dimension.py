import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from data import make_dataset
from features import build_features
from model import build_models
from utils.utils import save_results
import tensorflow as tf
import yaml
from yaml import Loader
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import mendeleev
import pathlib
import mlflow

def latent_parser():
    with open(str(pathlib.Path(__file__).absolute().parent.parent.parent) + '/config/available_model_config.yaml') as file:
        model_config = yaml.load(file, Loader)

    my_parser = argparse.ArgumentParser(prog='latent_dimension',
                                        description="train and display latent space of linear model",
                                        usage='%(prog)s [options]',
                                        )
    my_parser.add_argument('-model',
                           action='store',
                           nargs='+',
                           metavar='MODEL',
                           help="""Select the linear model. Possible choice:
                                  -linear regressor
                                  -linear classifier"""
                           )

    my_parser.add_argument('-config',
                           action='store',
                           nargs=1,
                           metavar='CONFIG',
                           help="""Use a custom config for the ML model.
                                   The model need to be specified"""
                           )

    my_parser.add_argument('--no-save',
                           action='store',
                           nargs='?',
                           help="Don't save/track results with mlflow. Arguments specify what to not track",
                           choices=['model', 'all', 'latent'],
                           const='all'
                           )


    # Parse arguments from cli
    args = my_parser.parse_args()

    return args


def main():
    with open(str(pathlib.Path(__file__).absolute().parent.parent.parent) +
              '/config/latent_dimension_config.yaml') as file:
        model_config = yaml.load(file, Loader)

    len_argv = len(sys.argv)
    model_config_path = None

    if len_argv > 1:
        args = latent_parser()
        if args.model is not None:
            model_name = ' '.join(args.model)
        else:
            model_name = 'linear regressor'
        model_config_path = args.config
        if args.no_save == 'all':
            disable_autolog = True
            log_models = False
            log_latent = False
        elif args.no_save == 'model':
            disable_autolog = False
            log_models = False
            log_latent = False
        elif args.no_save == 'latent':
            disable_autolog = False
            log_models = True
            log_latent = False
        else:
            disable_autolog = False
            log_models = True
            log_latent = True

    else:
        model_name = 'linear regressor'
        disable_autolog = False
        log_models = True
        log_latent = True

    if 'regressor' in model_name:
        # Load SuperCon dataset
        sc_dataframe = make_dataset.get_supercon(name='supercon.csv')
        tc = sc_dataframe['critical_temp']
        # Select material with high temperature (tc > 10 K)
        mask_temperature_materials = np.where(tc > 10, 1, 0)
        legend_latent_space = ['High temperature', 'Low temperature']

    elif 'classifier' in model_name:
        # Load SuperCon dataset
        sc_dataframe = make_dataset.get_supercon(name='supercon_garbage_50k.csv')
        tc = sc_dataframe['critical_temp'].apply(lambda x: int(x > 0))
        mask_temperature_materials = tc
        legend_latent_space = ['Superconductor', 'Non Superconductor']

    if model_config_path is not None:
        model_config_path = model_config_path[0]
    else:
        model_config_path = '{0}/config/latent_dimension_config.yaml'.format(
            str(pathlib.Path(__file__).absolute().parent.parent.parent))

    with open(model_config_path) as file:
        model_config = yaml.load(file, Loader)

    mlflow.set_tracking_uri(str(pathlib.Path(__file__).absolute().parent.parent.parent) + "/data/experiments/mlruns")
    mlflow.set_experiment('Latent' + model_name)

    # Load atomic data
    ptable = make_dataset.get_periodictable()
    # Initialize the processor for atomic data
    atom_processor = build_features.AtomData(ptable)
    # Process atomic data
    atom_processed = atom_processor.get_atom_data()

    # Initialize processor for SuperCon data
    supercon_processor = build_features.SuperConData(atom_processed, sc_dataframe, padding=10)
    # Process SuperCon data
    supercon_processed = supercon_processor.get_dataset()

    X, X_test, Y, Y_test = build_features.train_test_split(supercon_processed, tc,
                                                           model_config['train setup']['test split'])
    X, X_val, Y, Y_val = build_features.train_test_split(X, Y, model_config['train setup']['validation split'])

    with mlflow.start_run():
        # Define model and train it
        model = build_models.get_model(model_name=model_name, model_config=model_config)
        callbacks = [tf.keras.callbacks.EarlyStopping(**model_config['train setup']['early stopping setup'])]

        print(f"disable autolog : {disable_autolog}\nlog models : {log_models}\nlog latent : {log_latent}")

        # Logs metrics, params, model
        mlflow.tensorflow.autolog(disable=disable_autolog, log_models=log_models)

        model.fit(X, Y, validation_data=(X_val, Y_val), callbacks=callbacks, **model_config['train setup']['fit setup'])

        score = model.evaluate(X_test, Y_test, verbose=0)
        print(score)

        materials_representation = pd.DataFrame(model.material_representation(supercon_processed).numpy(),
                                                columns=['latent feature'])

        elements = [mendeleev.element(i).symbol for i in range(1, 97)]

        elements_dataframe = pd.DataFrame(np.eye(96), columns=elements)
        atoms_processor = build_features.SuperConData(atom_processed, elements_dataframe)
        # Process SuperCon data
        atoms_processed = atoms_processor.get_dataset()

        atoms_representation = pd.DataFrame(model.atom_representation(atoms_processed).numpy(), index=elements,
                                            columns=['latent feature'])

        if log_latent:
            # Save predictions of the model
            artifact_uri = mlflow.get_artifact_uri()
            atoms_representation.to_csv(artifact_uri + "/elements.csv", index=False)
            materials_representation.to_csv(artifact_uri + "/materials.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 8))

    sns.histplot(data=materials_representation, x='latent feature', hue=mask_temperature_materials, ax=axs[0])
    axs[0].set_title('Material Latent Representation')
    axs[0].set_xlabel('Latent feature')
    axs[0].legend(legend_latent_space)

    sns.scatterplot(data=atoms_representation, x='latent feature', y=range(1, 97), ax=axs[1])
    axs[1].set_title('Atomic Latent Representation')
    axs[1].set_xlabel('Latent feature')
    axs[1].set_ylabel('Atomic number')
    axs[1].set_yticks([])

    for line in range(0, atoms_representation.shape[0]):
        axs[1].text(x=atoms_representation.iloc[line] + 0.1 * atoms_representation.iloc[line],
                    y=line + 1,
                    s=elements[line],
                    fontsize=8,
                    )

    plt.show()
    print(atoms_representation)


#    if args is not None and args.save is not None:
#        save_results(score, model, [Y_test, model.predict(X_test)], args.save, atoms_representation,
#                     materials_representation)


if __name__ == '__main__':
    main()
