import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
from features import CategoricalFeature


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


def plot_features(data, features):
    for feature in features:
        ax = data[feature].plot.hist(bins=100)
        plt.clf()


def test_with_clf(train_x, train_y, test_x, test_y, features, feature_map, clf):
    all_features = []
    for feature in features:
        if isinstance(feature_map[feature], CategoricalFeature):
            all_features.extend(feature_map[feature].sub_features)
        else:
            all_features.append(feature)

    clf.fit(train_x[all_features], train_y)
    y_pred = clf.predict(test_x[all_features])
    res = sum([1 for i in range(len(y_pred)) if y_pred[i] == train_y.iloc[i]])/len(y_pred)
    return res
