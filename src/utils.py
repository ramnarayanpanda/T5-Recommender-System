import pickle
import gzip
import json


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_json_gz(jsonfilename):
    with gzip.open(jsonfilename, 'r') as fin:
        data = json.loads(fin.read().decode('utf-8'))
    return data

def flatten_dict(dct):
    list_items = {k: v for k, v in dct.items() if isinstance(v, list)}
    non_list_items = {k: v for k, v in dct.items() if not isinstance(v, list)}
    result = [
        {**non_list_items, **{k: v[i] for k, v in list_items.items()}}
        for i in range(len(next(iter(list_items.values()))))
    ]
    return result