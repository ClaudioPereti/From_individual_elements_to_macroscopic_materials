import sys
import mlflow
import pandas as pd
import pathlib

def load_classifier():
    print("OK")

import os
path = str(pathlib.Path(__file__).parent) + '/ensemble_classifier'

print(path)
list_dir = os.listdir(path)
list_dir.remove("meta.yaml")
print(len(list_dir))


load_classifier()






