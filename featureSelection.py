import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from utils import data_get_label

FILTER_THRESHOLD = 1


def mutal_information_filter(data, feature_list):
    print(f' Computing Mutual Information Scores')
    scores = np.zeros((len(feature_list), len(feature_list)))
    for i, feature in enumerate(feature_list):
        for j, feature2 in enumerate(feature_list):
            scores[i][j] = normalized_mutual_info_score(data[feature], data[feature2]) if i != j else 0
    print(f' final scores:\n{scores}\n end')
    return data, feature_list


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
        res_for_k.append(new_score, features)
        k += 1
    print('Feature sets and scores for iterative best k features:')
    print('K,Score,Features')
    for i in len(res_for_k):
        print('{},{},{}'.format(i+1,res_for_k[i][0], ','.join(res_for_k[i][1])))




def bds(data, features, clf):  # free palestine
    X, y = data_get_label(data)
    selected_features = []
    scores = []
    back_selected_features = features.copy()
    front_last_score = 0
    back_last_score = 0

    while not set(selected_features) == set(back_selected_features):
        forward_best_score = 0
        best_feature = None
        backward_best_score = 0
        worst_feature = None

        # Forward search
        for feature in back_selected_features:
            temp_features = selected_features.copy().append(feature)
            test_data = X[temp_features]
            score = cross_val_score(estimator=clf, X=test_data, y=y, cv=5).mean()
            if score > forward_best_score:
                forward_best_score = score
                best_feature = feature

        # Backward search
        for feature in back_selected_features:
            if feature in selected_features:
                continue
            temp_features = back_selected_features.copy().remove(feature)
            test_data = X[temp_features]
            score = cross_val_score(estimator=clf, X=test_data, y=y, cv=5).mean()
            if score > backward_best_score:
                backward_best_score = score
                worst_feature = feature

        # Which direction improves more
        # If stuck in local minimum: force forward search
        if (forward_best_score < front_last_score and back_last_score < back_last_score) or \
                (forward_best_score - front_last_score > backward_best_score - back_last_score):
            front_last_score = forward_best_score
            selected_features.append(best_feature)
            scores.append(forward_best_score)
        else:
            back_last_score = backward_best_score
            back_selected_features.remove(worst_feature)
    ret_val = [(selected_features[i], score[i]) for i in len(selected_features)]
    ret_val.sort(key=lambda x: x[1], reverse=True)
    print('Features and scores for Bi-directional search:')
    print('Feature,Score')
    for feature in ret_val:
        print('{},{}'.format(feature[0], feature[1]))
