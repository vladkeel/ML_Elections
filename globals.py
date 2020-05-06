categorical_features = ['Age_group', 'Looking_at_poles_results', 'Gender', 'Married', 'Voting_Time',
                        'Will_vote_only_large_party', 'Most_Important_Issue', 'Main_transportation',
                        'Occupation', 'Financial_agenda_matters']

discrete_features = ['Occupation_Satisfaction', 'Yearly_IncomeK', 'Last_school_grades',
                     'Number_of_differnt_parties_voted_for', 'Number_of_valued_Kneset_members',
                     'Num_of_kids_born_last_10_years']


party_lables = {'Yellows': 0, 'Whites': 1, 'Violets': 2, 'Turquoises': 3, 'Reds': 4, 'Purples': 5, 'Pinks': 6,
                'Oranges': 7, 'Khakis': 8, 'Greys': 9, 'Greens': 10, 'Browns': 11, 'Blues': 12}

age_groups = {'Below_30': 0, '30-45': 1, '45_and_up': 2}

bool_lables = {'No': 0, 'Yes': 1}

will_vote_lables = {'No': 0, 'Yes': 1, 'Maybe': 2}

gender_lables = {'Female':0, 'Male':1}

time_lables = {'By_16:00':0, 'After_16:00':1}

issues_lables = {'Education':1, 'Environment':2, 'Financial':3, 'Foreign_Affairs':4,
                 'Healthcare':5, 'Military':6, 'Social':7, 'Other':0}

transport_lables = {'Public_or_other':0, 'Car':1, 'Motorcycle_or_truck':2, 'Foot_or_bicycle':3}

occupation_lables = {'Industry_or_other':0, 'Services_or_Retail':1, 'Public_Sector':2,
                     'Student_or_Unemployed':3, 'Hightech':4}

translator = {'Vote': party_lables, 'Age_group': age_groups, 'Looking_at_poles_results': bool_lables,
              'Gender': gender_lables, 'Married': bool_lables, 'Voting_Time': time_lables,
              'Will_vote_only_large_party': will_vote_lables, 'Most_Important_Issue': issues_lables,
              'Main_transportation': transport_lables, 'Occupation': occupation_lables,
              'Financial_agenda_matters': bool_lables}

