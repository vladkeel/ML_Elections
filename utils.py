import numpy as np
from enum import Enum

class FeatureType(Enum):
    NOMINAL = 1
    CATEGORICAL = 2

class Feature:
    def __init__(self, f_type):
        feature_type = f_type

class CategoricalFeature(Feature):
    def __init__(self, f_type, feature):
        Feature.__init__(self, f_type)
        self.values_list = feature.unique()
        self.common = feature.mode()[0]

class NominalFeature(Feature):
    def __init__(self, f_type, feature):
        self.mean = feature.mean()
        self.median = feature.median()
        self.Q1 = feature.quantile(0.25)
        self.Q3 = feature.quantile(0.75)
        self.IQR = self.Q3 - self.Q1
        self.LF = self.Q1 - 1.5 * self.IQR
        self.UF = self.Q3 + 1.5 * self.IQR


