from features import FeatureType, CategoricalFeature, NominalFeature
import globals
import numpy as np
import pandas as pd
import sklearn as sk

FILTER_THRESHOLD = 1


def mutal_information_filter(data, feature_list):
    print(f' Computing Mutual Information Scores')
    scores = np.zeros((len(feature_list), len(feature_list)))
    for i, feature in enumerate(feature_list):
        for j, feature2 in enumerate(feature_list):
            scores[i][j] = sk.metrics.normalized_mutual_info_score(data[feature], data[feature2]) if i != j else 0
    print(f' final scores:\n{scores}\n end')
    return data, feature_list
