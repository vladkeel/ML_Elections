import pandas as pd
from random import shuffle


def data_get_label(data):
    return data.drop('Vote', axis=1), data['Vote']


def load_data(filename):
    data = pd.read_csv(filename, header=0)
    return data


def split_data(data):
    data_size = len(data.index)
    indices = list(range(data_size))
    shuffle(indices)
    train = data.iloc[indices[:int(data_size*0.75)], :]
    val = data.iloc[indices[int(data_size*0.75):int(data_size*0.9)], :]
    test = data.iloc[indices[int(data_size*0.9):], :]
    return train, val, test

def print_to_file(str, file):
    with open(file, 'a') as f:
        print(str, file=f)
