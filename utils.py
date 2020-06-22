import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
from features import CategoricalFeature
from sklearn.metrics import silhouette_score


def data_get_label(data):
    return data.drop('Vote', axis=1), data['Vote']


def load_data(filename):
    data = pd.read_csv(filename, header=0)
    return data


def save_data(data, filename):
    data.to_csv(filename, index=False)


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


def create_coalition(data, predicted_cluster):
    last_predicted = predicted_cluster.copy()
    last_score = -2
    max_score = -1
    while max_score > last_score:
        last_score = max_score
        max_predicted = None
        for i in range(max(last_predicted)):
            for j in range(max(last_predicted)):
                if i > j:
                    new_prediction = connect_clusters(last_predicted, i, j)
                    new_score = silhouette_score(data, new_prediction)
                    if new_score > max_score:
                        max_score = new_score
                        max_predicted = new_prediction
        if max_score > last_score:
            last_predicted = max_predicted
    return last_predicted


# assumption i > j
def connect_clusters(predicted_cluster, i, j):
    predicted_copy = predicted_cluster.copy()
    predicted_copy[predicted_copy == i] = j
    if i != max(predicted_copy):
        predicted_copy[predicted_copy == max(predicted_copy)] = i
    return predicted_copy
