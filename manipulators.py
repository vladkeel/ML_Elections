from features import CategoricalFeature, NominalFeature
import globals
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import normalize

NUMBER_OF_NEIGHBORS = 10

map_of_knns = {}


def fill_row(row, target, features, feature_map):
    if not pd.isna(row[target]):
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
        print(f' Making knn for feature {target} number {i} of {num}')
        origs = feature_list.copy()
        origs.remove(target)
        if isinstance(feature_map[target], CategoricalFeature):
            map_of_knns[target] = KNeighborsClassifier(n_neighbors=NUMBER_OF_NEIGHBORS)
        else:
            map_of_knns[target] = KNeighborsRegressor(n_neighbors=NUMBER_OF_NEIGHBORS)
        map_of_knns[target].fit(train_data[origs], train_data[target])

    for i, target in enumerate(feature_list):
        print(f' Completing feature {target} number {i} of {num}')
        origs = feature_list.copy()
        origs.remove(target)
        data.apply(fill_row, axis=1, target=target, features=origs, feature_map=feature_map)
        # data[target].fillna(fill_row(data[origs], target, origs, feature_map))
    return data


def clip_outliers(data, features, feature_map):
    for feature in features:
        if isinstance(feature_map[feature], NominalFeature):
            data[feature] = data[feature].clip(lower=feature_map[feature].LF, upper=feature_map[feature].UF)
        # else:
        #     data = data.set_index(feature)
        #     data[feature] = data.drop()
    return data


def enumerate_attrs(data):
    for feature in globals.categorical_features:
        data[feature] = data[feature].map(globals.translator[feature])
    return data


def normalize_features(data, features, feature_map):
    for feature in features:
        if isinstance(feature_map[feature], CategoricalFeature):
            continue
        data[feature] = normalize(data[feature], norm='l2')
