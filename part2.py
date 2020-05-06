from features import map_features
import featureSelection
import manipulators
from utils import data_get_label, load_data, split_data, plot_features, test_with_clf
from sklearn.ensemble import RandomForestClassifier
from heatmap import heatmap, annotate_heatmap
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def main():
    raw_data = load_data('ElectionsData.csv')
    data = raw_data.copy()
    #raw data saved aside for future inspection
    raw_train, raw_val, raw_test = split_data(data)
    train, val, test = raw_train.copy(), raw_val.copy(), raw_test.copy()
    train_x, train_y = data_get_label(train)
    val_x, val_y = data_get_label(val)
    test_x, test_y = data_get_label(test)

    train_x = manipulators.enumerate_attrs(train_x)
    val_x = manipulators.enumerate_attrs(val_x)
    test_x = manipulators.enumerate_attrs(test_x)

    train_y = manipulators.enumerate_lable(train_y)
    val_y = manipulators.enumerate_lable(val_y)
    test_y = manipulators.enumerate_lable(test_y)

    features_map = map_features(train_x)
    features = list(features_map.keys())

    #plot_features(train_x, features)

    filler = manipulators.NAFiller()
    filler.fit(train_x, features, features_map)

    train_x_c = filler.transform(train_x, features, features_map)
    val_x_c = filler.transform(val_x, features, features_map)
    test_x_c = filler.transform(test_x, features, features_map)

    normalizer = manipulators.DataNormalizer()
    normalizer.fit(train_x, features, features_map)

    train_x = normalizer.transform(train_x_c, features, features_map)
    val_x = normalizer.transform(val_x_c, features, features_map)
    test_x = normalizer.transform(test_x_c, features, features_map)

    one_hot = manipulators.OneHot()
    one_hot.fit(train_x, features, features_map)

    train_x = one_hot.transform(train_x, features, features_map)
    val_x = one_hot.transform(val_x, features, features_map)
    test_x = one_hot.transform(test_x, features, features_map)

    #mi_scores, mi_features = featureSelection.new_mutual_info_filter(test_x_c, features, features_map)
    #fig, ax = plt.subplots()
    #im, cbar = heatmap(mi_scores, features, features, ax=ax,
    #                   cmap="YlGn", cbarlabel="MI Score")
    #texts = annotate_heatmap(im, valfmt="{x:.1f} t")
    #fig.tight_layout()
    #plt.show()

    #test_x_n, feature_list, features_map = featureSelection.mutal_information_filter(test_x_n, features, features_map)
    #sfs with knn
    sfs_knn_features = featureSelection.sfs(train_x, train_y, features, features_map, KNeighborsClassifier(n_neighbors=5), 'sfs_knn.csv')
    #sfs with svm
    sfs_svm_features = featureSelection.sfs(train_x, train_y, features, features_map, RandomForestClassifier(n_estimators=100), 'sfs_forest.csv')
    k_best = featureSelection.select_k_best(train_x, train_y, 15)

    #Test the features:
    clf = KNeighborsClassifier(n_neighbors=5)
    #r_mi = test_with_clf(train_x, train_y, val_x, val_y, mi_features, features_map, clf)
    r_sfs_knn = test_with_clf(train_x, train_y, val_x, val_y, sfs_knn_features, features_map, clf)
    r_sfs_forest = test_with_clf(train_x, train_y, val_x, val_y, sfs_svm_features, features_map, clf)
    #r_k_best = test_with_clf(train_x, train_y, val_x, val_y, k_best, features_map, clf)
    #print('For mutual information filtering: {}'.format(r_mi))
    print('SFS with KNN: {}'.format(r_sfs_knn))
    print('SFS with random forest: {}'.format(r_sfs_forest))
    #print('select K best (15): {}'.format(r_k_best))


if __name__ == '__main__':
    main()
