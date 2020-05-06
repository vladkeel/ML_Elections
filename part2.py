from features import map_features
import featureSelection
import manipulators
from utils import data_get_label, load_data, split_data
from sklearn.ensemble import RandomForestClassifier


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
    featureSelection.bds(data, features, features_map, RandomForestClassifier(n_estimators=100))
    featureSelection.iterative_k_best(data, features_map, RandomForestClassifier(n_estimators=100))

    raw_train, raw_val, raw_test = split_data(data)
    train, val, test = raw_train.copy(), raw_val.copy(), raw_test.copy()
    train_x, train_y = data_get_label(train)
    val_x, val_y = data_get_label(val)
    test_x, test_y = data_get_label(test)
    print(train_x)
    a = 0


if __name__ == '__main__':
    main()