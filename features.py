import numpy as np
from pandas.api.types import is_numeric_dtype
from enum import Enum
import globals


class FeatureType(Enum):
    NOMINAL = 1
    CATEGORICAL = 2


class Feature:
    def __init__(self, f_type):
        self.feature_type = f_type


class CategoricalFeature(Feature):
    def __init__(self, feature):
        super(CategoricalFeature, self).__init__(FeatureType.CATEGORICAL)
        self.values_list = feature.unique()
        self.common = feature.mode()[0]

    def replacement(self):
        return self.common


class NominalFeature(Feature):
    def __init__(self, feature):
        super(NominalFeature, self).__init__(FeatureType.NOMINAL)
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
        return CategoricalFeature(feature)
    else:
        return NominalFeature(feature)


def map_features(data):
    feature_map = {}
    features = list(data.columns)
    for feature in features:
        feature_map[feature] = map_feature(feature, data[feature])
    return feature_map



