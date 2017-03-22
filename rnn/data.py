import yaml
import csv
import numpy as np

def create_feed(_data, conf):
    def create_row(_data, i, steps):
        return _data[i:i+steps]
    x = [create_row(_data, i, conf.time_steps) for i in
         range(len(_data) - conf.time_steps - conf.n_output_dim + 1)]
    y = [create_row(_data, i, conf.n_output_dim) for i in
         range(conf.time_steps, len(_data) - conf.n_output_dim + 1)]
    return x, np.asarray(y).reshape(-1, conf.n_output_dim)

def create_labelled_feed(_data, labels, conf):
    y = labels[conf.time_steps:].astype(np.float)
    def create_row(_data, i):
        return _data[i:i + conf.time_steps]
    x = [create_row(_data, i) for i in range(len(_data) - conf.time_steps)]
    return x, y

def load_file(path):
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        return np.asarray([data for data in reader])

def load_conf(path):
    with open(path, 'r') as stream:
        return yaml.load(stream)


class Config(object):
    pass
