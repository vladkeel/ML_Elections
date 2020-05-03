import pandas as pd
import numpy as np
import random
import utils
import globals

def load_data(filename):
    data = pd.read_csv(r'ElectionsData.csv', header=0)
    return data

def split_data(data):
    data_size = len(data.index)
    indices = list(range(data_size))
    random.shuffle(indices)
    train = data.iloc[indices[:int(data_size*0.75)],:]
    val = data.iloc[indices[int(data_size*0.75):int(data_size*0.9)],:]
    test = data.iloc[indices[int(data_size*0.9):],:]
    return train, val, test

def enumerate_attrs(data):
    #data['Lable'] = data['Vote'].map(globals.party_lables)
    data['Age'] = data['Age_group'].map(globals.age_groups)
    data['Look_at_polls'] = data['Looking_at_poles_results'].map(globals.bool_lables)
    data['Sex'] = data['Gender'].map(globals.gender_lables)
    data['Marital'] = data['Married'].map(globals.bool_lables)
    data['Late_vote'] = data['Voting_Time'].map(globals.time_lables)
    data['Big_party'] = data['Will_vote_only_large_party'].map(globals.bool_lables)
    data['Main_issue'] = data['Most_Important_Issue'].map(globals.issues_lables)
    data['Transport'] = data['Main_transportation'].map(globals.transport_lables)
    data['work'] = data['Occupation'].map(globals.occupation_lables)
    data['Financial_agenda'] = data['Financial_agenda_matters'].map(globals.bool_lables)
    data.drop(['Age_group', 'Looking_at_poles_results', 'Gender', 'Married', 'Voting_Time',
               'Will_vote_only_large_party', 'Most_Important_Issue', 'Main_transportation', 'Occupation',
               'Financial_agenda_matters'], axis=1)
    return data




def data_get_lable(data):
    return data.drop('Vote', axis=1), data['Vote'].map(globals.party_lables)



def main():
    data = load_data('ElectionsData.csv')
    #raw data saved aside for future inspection
    features_map = utils.map_features(data)
    raw_train, raw_val, raw_test = split_data(data)
    train, val, test = raw_train.copy(), raw_val.copy(), raw_test.copy()
    train_x, train_y = data_get_lable(train)
    val_x, val_y = data_get_lable(val)
    test_x, test_y = data_get_lable(test)
    train_x = enumerate_attrs(train_x)
    val_x = enumerate_attrs(val_x)
    test_x = enumerate_attrs(test_x)
    print(train_x)
    a = 0


if __name__ == '__main__':
    main()