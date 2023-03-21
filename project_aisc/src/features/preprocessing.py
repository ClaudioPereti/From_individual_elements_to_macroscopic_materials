import pandas as pd

from data import make_dataset
from features import build_features
import yaml
from yaml import Loader
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split
import json


def preprocess_data(problem, supercon_data, garbagein_data, test_split, val_split, other_data=None, seed=42,
                    padding=10):

    home_path = str(pathlib.Path(__file__).absolute().parent.parent.parent)
    # Load atomic data
    ptable = make_dataset.get_periodictable()
    # Initialize the processor for atomic data
    atom_processor = build_features.AtomData(ptable)
    # Process atomic data
    atom_processed = atom_processor.get_atom_data()

    if problem == 'regression':
        sc_dataframe = make_dataset.get_supercon(name=supercon_data)
        tc = sc_dataframe['critical_temp']
    elif problem == 'classification':
        supercon_dataframe = make_dataset.get_supercon(name=supercon_data)
        garbagein_dataframe = make_dataset.get_supercon(name=garbagein_data)
        tc_supercon = np.ones(supercon_dataframe['critical_temp'].shape[0])
        tc_garbage = np.zeros(garbagein_dataframe['critical_temp'].shape[0])
        # Merge supercondutors data non-superconductors data
        sc_dataframe = pd.concat([supercon_dataframe, garbagein_dataframe], ignore_index=True)
        tc = np.concatenate([tc_supercon, tc_garbage])

    # Initialize processor for SuperCon data
    supercon_processor = build_features.SuperConData(atom_processed, sc_dataframe, padding=padding)
    # Process SuperCon data
    supercon_processed = supercon_processor.get_dataset()

    X, X_test, Y, Y_test = build_features.train_test_split(supercon_processed, tc, test_split, seed)
    X, X_val, Y, Y_val = build_features.train_test_split(X, Y, val_split, seed)

    np.save(home_path + '/data/processed/train/X_train.npy', np.array(X))
    np.save(home_path + '/data/processed/train/Y_train.npy', np.array(Y))
    np.save(home_path + '/data/processed/val/X_val.npy', np.array(X_val))
    np.save(home_path + '/data/processed/val/Y_val.npy', np.array(Y_val))
    np.save(home_path + '/data/processed/test/X_test.npy', np.array(X_test))
    np.save(home_path + '/data/processed/test/Y_test.npy', np.array(Y_test))


    index_dataset = [i for i in tc.index]
    index_train, index_test = sk_train_test_split(index_dataset, test_size=test_split, random_state=seed)
    index_train, index_val = sk_train_test_split(index_train, test_size=test_split , random_state=seed)
    index_data = {'seed': seed,
                  'index_train': index_train,
                  'index_val': index_val,
                  'index_test': index_test}

    with open(home_path + '/../index_train_test_data.json', 'w') as file:
        json.dump(index_data, file)

    if other_data is not None:
        for data_name in other_data:
            external_dataset = make_dataset.get_supercon(name="../external/" + data_name)
            tc = None
            if 'critical_temp' in external_dataset.columns:
                tc = external_dataset['critical_temp']
            supercon_processor = build_features.SuperConData(atom_processed, external_dataset, padding=padding)
            supercon_processed = supercon_processor.get_dataset()
            np.save(home_path + '/data/processed/' + data_name.replace('csv', 'npy'), np.array(supercon_processed))
            if tc is not None:
                np.save(home_path + '/data/processed/Y_' + data_name.replace('csv', 'npy'), np.array(tc))


if __name__ == '__main__':
    preprocessing_config_path = str(pathlib.Path(__file__).absolute().parent.parent.parent) + \
                                '/config/preprocessing.yaml'

    with open(preprocessing_config_path) as file:
        preprocessing_config = yaml.load(file, Loader)

    preprocess_data(**preprocessing_config)
