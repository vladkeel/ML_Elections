from features import map_features
import featureSelection
import manipulators
from utils import data_get_label, load_data, split_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


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

    filler = manipulators.NAFiller()
    filler.fit(train_x, features, features_map)

    train_x = filler.transform(train_x, features, features_map)
    val_x = filler.transform(val_x, features, features_map)
    test_x = filler.transform(test_x, features, features_map)

    normalizer = manipulators.DataNormalizer()
    normalizer.fit(train_x, features, features_map)

    train_x = normalizer.transform(train_x, features, features_map)
    val_x = normalizer.transform(val_x, features, features_map)
    test_x = normalizer.transform(test_x, features, features_map)

    one_hot = manipulators.OneHot()
    one_hot.fit(train_x, features, features_map)

    train_x = one_hot.transform(train_x, features, features_map)
    val_x = one_hot.transform(val_x, features, features_map)
    test_x = one_hot.transform(test_x, features, features_map)



    #sfs with knn
    sfs_knn_features = featureSelection.sfs(train_x, train_y, features, features_map, KNeighborsClassifier(n_neighbors=5), 'sfs_knn.csv')
    #sfs with svm
    #sfs_svm_features = featureSelection.sfs(train_x, train_y, features, features_map, RandomForestClassifier(n_estimators=100), 'sfs_forest.csv')
    #featureSelection.iterative_k_best(data, RandomForestClassifier(n_estimators=100))

    a = 0


if __name__ == '__main__':
    main()
