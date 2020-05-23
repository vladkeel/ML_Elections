from features import map_features
import manipulators
from utils import data_get_label, load_data, split_data, save_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

useful_features = ['Vote', 'Yearly_IncomeK', 'Number_of_differnt_parties_voted_for', 'Political_interest_Total_Score',
                   'Avg_Satisfaction_with_previous_vote', 'Avg_monthly_income_all_years', 'Most_Important_Issue',
                   'Overall_happiness_score', 'Avg_size_per_room, Weighted_education_rank']

features_map = None


def load_and_select():
    raw_data = load_data('ElectionsData.csv')
    data = raw_data.copy()
    data = data[useful_features]
    # raw data saved aside for future inspection
    raw_train, raw_val, raw_test = split_data(data)
    train, val, test = raw_train.copy(), raw_val.copy(), raw_test.copy()
    train_x, train_y = data_get_label(train)
    val_x, val_y = data_get_label(val)
    test_x, test_y = data_get_label(test)
    return train_x, train_y, val_x, val_y, test_x, test_y


def prepare_data(train_x, train_y, val_x, val_y, test_x, test_y):
    global features_map
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

    return train_x, train_y, val_x, val_y, test_x, test_y


def main():
    train_x, train_y, val_x, val_y, test_x, test_y = load_and_select()
    train_x, train_y, val_x, val_y, test_x, test_y = prepare_data(train_x, train_y, val_x, val_y, test_x, test_y)

    save_data(train_x, 'train_x.csv')
    save_data(train_y, 'train_y.csv')
    save_data(val_x, 'val_x.csv')
    save_data(val_y, 'val_y.csv')
    save_data(test_x, 'test_x.csv')
    save_data(test_y, 'test_y.csv')

    # Make train here...
