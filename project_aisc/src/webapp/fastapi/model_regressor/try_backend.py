import requests
import json
import pandas as pd

if __name__ == "__main__":
    path = "http://0.0.0.0:8000/predict"

    data = {'formula': 'YBO'}
    data_json = json.dumps(data)
    #file = pd.read_csv('/home/claudio/AISC/project_aisc/data/raw/supercon.csv')['material'].to_list()
    #data = json.dumps({'file_formulas': file})
    #payload = {'formula_files': data}
    print(data_json)
    r = requests.post(path, json=data)

    print(r)
    print(r.json())

    data = {'formula': 'WC'}
    data_json = json.dumps(data)
    #file = pd.read_csv('/home/claudio/AISC/project_aisc/data/raw/supercon.csv')['material'].to_list()
    #data = json.dumps({'file_formulas': file})
    #payload = {'formula_files': data}
    print(data_json)
    r = requests.post(path, json=data)

    print(r)
    print(r.json())

    data = {'formula': 'Au0.978In0.022'}
    data_json = json.dumps(data)
    #file = pd.read_csv('/home/claudio/AISC/project_aisc/data/raw/supercon.csv')['material'].to_list()
    #data = json.dumps({'file_formulas': file})
    #payload = {'formula_files': data}
    print(data_json)
    r = requests.post(path, json=data)

    print(r)
    print(r.json())

    path = "http://0.0.0.0:8000/file"

    #data_json = json.dumps(data)
    file_data = pd.read_csv('/home/claudio/AISC/project_aisc/data/raw/supercon.csv')['material'].to_list()[:5]
    #data = json.dumps({'file_formulas': file})
    #payload = {'formula_files': data}
    data = {'many_formula': file_data}
    r = requests.post(path, json=data)

    print(r)
    print(r.json())


