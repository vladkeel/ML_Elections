from features import CategoricalFeature, NominalFeature
import globals
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import preprocessing

NUMBER_OF_NEIGHBORS = 20


class NAFiller:
    def __init__(self):
        self.map_of_knns = {}

    def fit(self, data, feature_list, feature_map):
        train_data = data.dropna()
        for i, target in enumerate(feature_list):
            origs = feature_list.copy()
            origs.remove(target)
            if isinstance(feature_map[target], CategoricalFeature):
                self.map_of_knns[target] = KNeighborsClassifier(n_neighbors=NUMBER_OF_NEIGHBORS)
            else:
                self.map_of_knns[target] = KNeighborsRegressor(n_neighbors=NUMBER_OF_NEIGHBORS)
            self.map_of_knns[target].fit(train_data[origs], train_data[target])

    def _fill_row(self, row, target, features, feature_map):
        if not np.isnan(row[target]):
            return row
        l_row = row.copy()
        for feature in features:
            if pd.isna(l_row[feature]):
                l_row[feature] = feature_map[feature].replacement()
        row[target] = self.map_of_knns[target].predict([l_row[features]])
        return row

    def transform(self, data, feature_list, feature_map):
        for i, target in enumerate(feature_list):
            #print(f'{i},', end='')
            origs = feature_list.copy()
            origs.remove(target)
            data = data.apply(self._fill_row, axis=1, target=target, features=origs, feature_map=feature_map)
        return data


class DataNormalizer:
    def __init__(self):
        self.scalers = {}

    def fit(self, data, features, feature_map):
        for feature in features:
            if isinstance(feature_map[feature], CategoricalFeature):
                continue
            x = data[[feature]].values.astype(float)
            self.scalers[feature] = preprocessing.MinMaxScaler()
            self.scalers[feature].fit(x)

    def transform(self, data, feature, feature_map):
        for feature in feature:
            if isinstance(feature_map[feature], CategoricalFeature):
                continue
            x = data[[feature]].values.astype(float)
            x_scaled = self.scalers[feature].transform(x)
            data[feature] = pd.DataFrame(x_scaled, columns=[feature], index=data.index)
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


def enumerate_lable(y):
    y = y.map(globals.translator['Vote'])
    return y


class OneHot:
    def __init__(self):
        self.encoders = {}

    def fit(self, data, features, feature_map):
        for feature in features:
            if isinstance(feature_map[feature], CategoricalFeature):
                self.encoders[feature] = preprocessing.OneHotEncoder(handle_unknown='ignore')
                self.encoders[feature].fit(data[[feature]])
                feature_map[feature].sub_features = self.encoders[feature].get_feature_names([feature])

    def transform(self, data, features, feature_map):
        for feature in features:
            if isinstance(feature_map[feature], CategoricalFeature):
                new_features = self.encoders[feature].transform(data[[feature]])
                new_names = feature_map[feature].sub_features
                for i, sub_feat in enumerate(new_names):
                    column = new_features[:,0].A.flatten()
                    data[sub_feat] = pd.DataFrame(column, columns=[sub_feat], index=data.index)
                data = data.drop(feature, axis=1)
        return data


