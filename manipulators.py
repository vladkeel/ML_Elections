from features import CategoricalFeature, NominalFeature
import globals
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import preprocessing

NUMBER_OF_NEIGHBORS = 200

map_of_knns = {}


def fill_row(row, target, features, feature_map):
    if not np.isnan(row[target]):
        return row
    l_row = row.copy()
    for feature in features:
        if pd.isna(l_row[feature]):
            l_row[feature] = feature_map[feature].replacement()
    row[target] = map_of_knns[target].predict([l_row[features]])
    return row


def fill_empty(data, feature_list, feature_map):
    train_data = data.dropna()
    print('Filling empty values')
    num = len(feature_list)

    for i, target in enumerate(feature_list):
        origs = feature_list.copy()
        origs.remove(target)
        if isinstance(feature_map[target], CategoricalFeature):
            map_of_knns[target] = KNeighborsClassifier(n_neighbors=NUMBER_OF_NEIGHBORS)
        else:
            map_of_knns[target] = KNeighborsRegressor(n_neighbors=NUMBER_OF_NEIGHBORS)
        map_of_knns[target].fit(train_data[origs], train_data[target])

    for i, target in enumerate(feature_list):
        print(f'{i},', end='')
        origs = feature_list.copy()
        origs.remove(target)
        data = data.apply(fill_row, axis=1, target=target, features=origs, feature_map=feature_map)
    return data


def clip_outliers(data, features, feature_map):
    for feature in features:
        if isinstance(feature_map[feature], NominalFeature):
            data[feature] = data[feature].clip(lower=feature_map[feature].LF, upper=feature_map[feature].UF)
    return data


def enumerate_attrs(data):
    for feature in globals.categorical_features:
        data[feature] = data[feature].map(globals.translator[feature])
    return data


def normalize_features(data, features, feature_map):
    for feature in features:
        if isinstance(feature_map[feature], CategoricalFeature):
            continue
        #data[feature] = data[feature] - data[feature].
        min_max_scaler = preprocessing.MinMaxScaler()
        x = data[[feature]].values.astype(float)
        x_scaled = min_max_scaler.fit_transform(x)
        data[feature] = pd.DataFrame(x_scaled, columns=[feature], index=data.index)
    return data


def one_hot(data, features, feature_map):
    for feature in features:
        if isinstance(feature_map[feature], CategoricalFeature):
            new_features = pd.get_dummies(data[feature], prefix=feature)
            new_names = new_features.columns
            feature_map[feature].sub_features = new_names
            for sub_feat in new_names:
                data[sub_feat] = new_features[sub_feat]
            data = data.drop(feature, axis=1)
    return data
