from enum import Enum
import globals


class FeatureType(Enum):
    NOMINAL = 1
    CATEGORICAL = 2


def remove_prefix(full, prx):
    if full.startswith(prx):
        return full[len(prx)+1:]


class Feature:
    def __init__(self, f_type, name):
        self.feature_type = f_type
        self.feature_name = name


class CategoricalFeature(Feature):
    def __init__(self, feature, name):
        super(CategoricalFeature, self).__init__(FeatureType.CATEGORICAL, name)
        self.values_list = feature.unique()
        self.common = feature.mode()[0]
        self.sub_features = []

    def replacement(self):
        return self.common

    def one_hot(self, series):
        for feat in series.columns:
            if series[feat] == 1:
                return remove_prefix(feat, super(CategoricalFeature, self).feature_name)


class NominalFeature(Feature):
    def __init__(self, feature, name):
        super(NominalFeature, self).__init__(FeatureType.NOMINAL, name)
        self.mean = feature.mean()
        self.median = feature.median()
        self.Q1 = feature.quantile(0.25)
        self.Q3 = feature.quantile(0.75)
        self.IQR = self.Q3 - self.Q1
        self.LF = self.Q1 - 1.5 * self.IQR
        self.UF = self.Q3 + 1.5 * self.IQR

    def replacement(self):
        return self.mean


def map_feature(name, feature):
    if name in globals.categorical_features:
        return CategoricalFeature(feature, name)
    else:
        return NominalFeature(feature, name)


def map_features(data):
    feature_map = {}
    features = list(data.columns)
    features.remove('Vote')
    for feature in features:
        feature_map[feature] = map_feature(feature, data[feature])
    return feature_map



