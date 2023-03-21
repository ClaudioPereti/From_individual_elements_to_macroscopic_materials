import numpy as np


def save_results(score=None, model=None, evaluations=None, arg_save=[], elements=None, materials=None):
    """Save results and model

    Save score, model and evaluations if specified through cli.
    Evaluations is a csv file containing an index, the observed critical temperature
    and the relative predictions. The index is referred to the selected materials.

    Args:
        - score: list containing model's metrics evaluated on test set
        - model: tf.keras.model that will be saved
        - evaluations: list containing Y_test and the relative predictions
        - arg_save: string or list to target what to save. It can contain:
           - 'all' -> save score, model, evaluations
           - 'score' -> save score
           - 'model' -> save model with SavedModel format
           - 'evaluations' -> save indexes, tests and predictions

    """
    import os
    import datetime
    import csv
    import pandas as pd

    date = datetime.datetime.now()

    directory = '/home/claudio/AISC/project_aisc/data/experiments/experiments_' + date.strftime(
        "%d") + "-" + date.strftime("%m")
    # Flag to check if an experiment and relative directory containing that data is alredy present
    today_experiments = os.path.isdir(directory)

    if not today_experiments:
        os.makedirs(directory)
    # count the number of experiments done for the day
    n_experiment_per_day = len([name for name in os.listdir(directory)])
    # each experiment has their own folder
    experiment_name = directory + '/experiment' + "-" + str(n_experiment_per_day)
    current_experiment = os.path.isdir(experiment_name)

    if not current_experiment:
        os.makedirs(experiment_name)

    if 'all' in arg_save or 'score' in arg_save:
        with open(experiment_name + '/score.csv', mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([metric.name for metric in model.metrics])
            csv_writer.writerow(score)

    if 'all' in arg_save or 'model' in arg_save:
        model.save(experiment_name + '/model')

    if 'all' in arg_save or 'test' in arg_save:
        ob_and_pred = pd.DataFrame(
            {'observed': evaluations[0].values, 'predicted': [value[0] for value in evaluations[1]]},
            index=evaluations[0].index)
        ob_and_pred.to_csv(experiment_name + '/evaluations.csv')
    if 'all' in arg_save or 'elements' in arg_save:
        try:
            elements.to_csv(experiment_name + '/elements.csv')
        except:
            pass
    if 'all' in arg_save or 'materials' in arg_save:
        try:
            materials.to_csv(experiment_name + '/materials.csv')
        except:
            pass


def weighted_average(iterable, weights):
    return np.average(iterable, weights=weights)


def geo_mean(iterable):
    iterable = np.abs(iterable)
    return iterable.prod() ** (1.0 / len(iterable))


def weighted_geo_mean(iterable, weights):
    iterable = np.abs(iterable) ** (weights / np.sum(weights))
    return iterable.prod() ** (1.0 / len(iterable))


def entropy(iterable):
    iterable = np.abs(iterable)
    iterable = np.where(iterable > 0.00000000001, iterable, 0.00000000001)
    return -np.sum(iterable * np.log(iterable))


def weighted_entropy(iterable, weights):
    iterable = np.abs(iterable)
    iterable = np.where(iterable > 0.00000000001, iterable, 0.00000000001)
    iterable = iterable * weights / np.sum(iterable * weights)

    return -np.sum(iterable * np.log(iterable))


def range_feature(iterable):
    max = iterable.max()
    min = iterable.min()
    return max - min


def weighted_range_feature(iterable, weights):
    iterable = iterable * weights / np.sum(weights)
    return range_feature(iterable)


def weighted_std(iterable, weights):
    iterable = abs(iterable - np.average(iterable, weights=weights)) ** 2
    std = np.sqrt(np.average(iterable, weights=weights))
    return std

class ModelNotFound(ValueError):
    """Model not found in config file: available_model_config.yaml"""
    pass

class ModelNotBuilt(ValueError):
    """Error in model building"""
    pass
