import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from utils import data_get_label, print_to_file
from features import CategoricalFeature, NominalFeature


FILTER_THRESHOLD = 0.99


def mutal_information_filter(data, feature_list, features_map):
    print_to_file(f' Computing Mutual Information Scores', 'mutual.txt')
    scores = np.zeros((len(feature_list), len(feature_list)))
    print_to_file(',{}'.format(','.join(feature_list)), 'mutual.txt')
    for i, feature in enumerate(feature_list):
        for j, feature2 in enumerate(feature_list):
            scores[i][j] = normalized_mutual_info_score(data[feature], data[feature2]) if i != j else 0
        print_to_file('{},{}'.format(feature_list[i], ','.join([str(x) for x in scores[i]])), 'mutual.txt')
    #print_to_file(f' final scores:\n{scores}\n end', 'mutual.txt')
    over_threshold = [sum([1 for i in j if i > FILTER_THRESHOLD]) for j in scores]
    while any(i > 0 for i in over_threshold):
        max_index = over_threshold.index(max(over_threshold))
        print_to_file(f' Deleting feature number {max_index} which is {feature_list[max_index]}', 'mutual.txt')
        scores = np.delete(scores, max_index, 0)
        scores = np.delete(scores, max_index, 1)
        data.drop(feature_list[max_index], axis=1)
        features_map.pop(feature_list[max_index])
        feature_list.pop(max_index)
        over_threshold = [sum(1 for i in j if i > FILTER_THRESHOLD) for j in scores]
    print_to_file(f' Feature list after filter {feature_list} for total of {len(feature_list)} features', 'mutual.txt')
    return data, feature_list, features_map


def select_k_best(X, y, k):
    features = list(X.columns)
    if len(features) <= k:
        return [(feature, 0) for feature in features]
    clf = SelectKBest(chi2, k=k)
    clf.fit(X,y)
    features_mask = clf.get_support()
    return [b for a, b in zip(features, features_mask) if a]


def iterative_k_best(data, clf, eps=1e-8):
    X, y = data_get_label(data)
    k = 1
    last_score = 0
    new_score = eps * 2
    res_for_k = []
    while new_score - last_score > eps:
        features = select_k_best(X, y, k)
        new_score = cross_val_score(estimator=clf, X=X, y=y, cv=5).mean()
        res_for_k.append((new_score, features))
        k += 1
    print_to_file('Feature sets and scores for iterative best k features:', 'itk.csv')
    print_to_file('K,Score,Features', 'itk.csv')
    for i in range(len(res_for_k)):
        print_to_file('{},{},{}'.format(i+1,res_for_k[i][0], ','.join(res_for_k[i][1])), 'itk.csv')


def sfs(X, y, features, feature_map, clf, outfile, eps=1e-8):
    selected_features = []
    selected_features_extended = []
    scores = []
    score_diff = 1
    last_score = 0
    epoch = 0
    print_to_file('Sequential forward selection search:', outfile)
    candidates = features.copy()
    while score_diff > eps:
        best_feature = None
        forward_best_score = 0
        epoch += 1
        # Forward search
        for feature in candidates:
            temp_features = selected_features_extended.copy()
            if isinstance(feature_map[feature], CategoricalFeature):
                temp_features.extend(feature_map[feature].sub_features)
            else:
                temp_features.append(feature)
            test_data = X[temp_features]
            score = cross_val_score(estimator=clf, X=test_data, y=y, cv=4).mean()
            if score > forward_best_score:
                forward_best_score = score
                best_feature = feature
        score_diff = forward_best_score - last_score
        if score_diff < eps:
            break
        last_score = forward_best_score
        selected_features.append(best_feature)
        scores.append(last_score)
        candidates.remove(best_feature)
        if isinstance(feature_map[best_feature], CategoricalFeature):
            selected_features_extended.extend(feature_map[best_feature].sub_features)
        else:
            selected_features_extended.append(best_feature)
        print(f'{epoch},', end='')
    ret_val = [(selected_features[i], scores[i]) for i in range(len(selected_features))]
    ret_val.sort(key=lambda x: x[1], reverse=True)
    print_to_file('Features and scores for Sequential forward search:', outfile)
    print_to_file('Feature,Score', outfile)
    for feature in ret_val:
        print_to_file('{},{}'.format(feature[0], feature[1]), outfile)

    return selected_features
