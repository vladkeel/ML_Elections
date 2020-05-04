import pandas as pd
import numpy as np
import random
from features import FeatureType, CategoricalFeature, NominalFeature, map_features
from featureSelection import mutal_information_filter
import globals
import manipulators


def load_data(filename):
    data = pd.read_csv(r'ElectionsData.csv', header=0)
    return data


def split_data(data):
    data_size = len(data.index)
    indices = list(range(data_size))
    random.shuffle(indices)
    train = data.iloc[indices[:int(data_size*0.75)], :]
    val = data.iloc[indices[int(data_size*0.75):int(data_size*0.9)], :]
    test = data.iloc[indices[int(data_size*0.9):], :]
    return train, val, test


def data_get_label(data):
    return data.drop('Vote', axis=1), data['Vote']


def main():
    raw_data = load_data('ElectionsData.csv')
    data = raw_data.copy()
    #raw data saved aside for future inspection
    data = manipulators.enumerate_attrs(data)
    features_map = map_features(data)
    features = list(features_map.keys())
    data = manipulators.fill_empty(data, features, features_map)
    data = manipulators.clip_outliers(data, features, features_map)
    mutal_information_filter(data, features)

    raw_train, raw_val, raw_test = split_data(data)
    train, val, test = raw_train.copy(), raw_val.copy(), raw_test.copy()
    train_x, train_y = data_get_label(train)
    val_x, val_y = data_get_label(val)
    test_x, test_y = data_get_label(test)
    print(train_x)
    a = 0


if __name__ == '__main__':
    main()