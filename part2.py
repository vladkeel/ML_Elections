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
    data = manipulators.enumerate_attrs(data)
    features_map = map_features(data)
    features = list(features_map.keys())
    data = manipulators.fill_empty(data, features, features_map)
    data = manipulators.clip_outliers(data, features, features_map)
    data = manipulators.normalize_features(data, features, features_map)
    data, features, features_map = featureSelection.mutal_information_filter(data, features, features_map)
    data = manipulators.one_hot(data, features, features_map)
    #sfs with knn
    sfs_knn_features = featureSelection.sfs(data, features, features_map, KNeighborsClassifier(n_neighbors=200), 'sfs_knn.csv')
    #sfs with svm
    sfs_svm_features = featureSelection.sfs(data, features, features_map, SVC(gamma='auto'), 'sfs_svm.csv')
    #featureSelection.iterative_k_best(data, RandomForestClassifier(n_estimators=100))

    raw_train, raw_val, raw_test = split_data(data)
    train, val, test = raw_train.copy(), raw_val.copy(), raw_test.copy()
    train_x, train_y = data_get_label(train)
    val_x, val_y = data_get_label(val)
    test_x, test_y = data_get_label(test)
    print(train_x)
    a = 0


if __name__ == '__main__':
    main()
