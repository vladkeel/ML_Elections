from features import map_features
import manipulators
from utils import data_get_label, load_data, split_data, save_data, create_coalition
import clustering
import globals
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

features_map = None
NUM_OF_PARTIES = len(globals.party_lables)


def load_and_select():
    raw_data = load_data('ElectionsData.csv')
    data = raw_data.copy()
    data = data[globals.useful_features]
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

    # training
    print(f'starting to train')

    #clustering
    print(f'clustering')
    gmm = clustering.GMM()
    gmm.fit(train_x, val_x)
    print(f'n_components {gmm.components} type {gmm.cvType}')
    prediction_gmm = gmm.predict(test_x)

    plt.figure(1)
    plt.scatter(test_x['Yearly_IncomeK'], test_y, c=prediction_gmm, s=50, cmap='viridis')
    plt.xlabel('Yearly_IncomeK')
    plt.ylabel('Vote')
    plt.title('Vote against Yearly Income', y=1.05)
    plt.show()

    plt.figure(2)
    plt.scatter(test_x['Number_of_differnt_parties_voted_for'], test_y, c=prediction_gmm, s=50, cmap='viridis')
    plt.xlabel('Number_of_differnt_parties_voted_for')
    plt.ylabel('Vote')
    plt.title('Vote against Number_of_differnt_parties_voted_for', y=1.05)
    plt.show()

    plt.figure(3)
    plt.scatter(test_x['Political_interest_Total_Score'], test_y, c=prediction_gmm, s=50, cmap='viridis')
    plt.xlabel('Political_interest_Total_Score')
    plt.ylabel('Vote')
    plt.title('Vote against Political_interest_Total_Score', y=1.05)
    plt.show()

    plt.figure(4)
    plt.scatter(test_x['Avg_Satisfaction_with_previous_vote'], test_y, c=prediction_gmm, s=50, cmap='viridis')
    plt.xlabel('Avg_Satisfaction_with_previous_vote')
    plt.ylabel('Vote')
    plt.title('Vote against Avg_Satisfaction_with_previous_vote', y=1.05)
    plt.show()

    plt.figure(5)
    plt.scatter(test_x['Avg_monthly_income_all_years'], test_y, c=prediction_gmm, s=50, cmap='viridis')
    plt.xlabel('Avg_monthly_income_all_years')
    plt.ylabel('Vote')
    plt.title('Vote against Avg_monthly_income_all_years', y=1.05)
    plt.show()

    plt.figure(6)
    plt.scatter(test_x['Overall_happiness_score'], test_y, c=prediction_gmm, s=50, cmap='viridis')
    plt.xlabel('Overall_happiness_score')
    plt.ylabel('Vote')
    plt.title('Vote against Overall_happiness_score', y=1.05)
    plt.show()

    plt.figure(7)
    plt.scatter(test_x['Avg_size_per_room'], test_y, c=prediction_gmm, s=50, cmap='viridis')
    plt.xlabel('Avg_size_per_room')
    plt.ylabel('Vote')
    plt.title('Vote against Avg_size_per_room', y=1.05)
    plt.show()

    plt.figure(8)
    plt.scatter(test_x['Weighted_education_rank'], test_y, c=prediction_gmm, s=50, cmap='viridis')
    plt.xlabel('Weighted_education_rank')
    plt.ylabel('Vote')
    plt.title('Vote against Weighted_education_rank', y=1.05)
    plt.show()

    silhouette_avg = silhouette_score(test_x, prediction_gmm)
    print(f'silhouette_avg {silhouette_avg}')
    predicted_coalition = create_coalition(test_x, prediction_gmm)
    silhouette_avg = silhouette_score(test_x, predicted_coalition)
    print(f'final silhouette_avg {silhouette_avg}')

    plt.figure(9)
    plt.scatter(test_x['Yearly_IncomeK'], test_y, c=predicted_coalition, s=50, cmap='viridis')
    plt.xlabel('Yearly_IncomeK')
    plt.ylabel('Vote')
    plt.title('Vote against Yearly Income', y=1.05)
    plt.show()


if __name__ == '__main__':
    main()
