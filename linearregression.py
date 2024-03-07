import pm4py as pm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
import numpy as np


def xes_to_df(file_path):
    event_log = pm.read_xes(file_path)
    event_df = pm.convert_to_dataframe(event_log)

    return event_df, event_log


def make_baseline_time(df_train):
    # get required columns and add time difference column
    df_sorted = df_train[['case:concept:name', 'concept:name', 'lifecycle:transition', 'time:timestamp']]
    df_sorted.sort_values(['case:concept:name', 'time:timestamp'], inplace=True)
    df_sorted['time_until_next'] = df_sorted['time:timestamp'].shift(-1) - df_sorted['time:timestamp']

    # add relative index
    label_encoder = LabelEncoder()
    df_sorted['encoded_concept:name'] = label_encoder.fit_transform(df_sorted['concept:name'])
    df_sorted['relative_index'] = df_sorted.groupby('case:concept:name').cumcount() + 1
    # remove worthless values
    df_sorted['time_until_next'] = df_sorted['time_until_next'].mask(
        df_sorted['time_until_next'] < pd.Timedelta("0 days"))

        # setup the dataframe
    average_df = pd.DataFrame()
    cases = ['A_SUBMITTED', 'A_PARTLYSUBMITTED', 'A_PREACCEPTED', 'W_Completeren aanvraag', 'A_ACCEPTED', 'O_SELECTED',
             'A_FINALIZED',
             'O_CREATED', 'O_SENT', 'O_SENT_BACK', 'W_Valideren aanvraag', 'A_REGISTERED', 'A_APPROVED', 'O_ACCEPTED',
             'W_Wijzigen contractgegevens', 'A_ACTIVATED', 'A_DECLINED', 'A_CANCELLED', 'W_Completeren aanvraag',
             'W_Nabellen incomplete dossiers', 'W_Afhandelen leads', 'W_Nabellen offertes', 'W_Beoordelen fraude']

    # get all the average times for the tasks
    for case in cases:
        temp_df = df_sorted.loc[df_sorted['concept:name'] == case].copy()
        temp_data = [[case, temp_df['time_until_next'].mean()]]
        average_df = average_df._append(temp_data, ignore_index=True)

    average_df.rename(columns={0: 'concept:name', 1: 'time_until_next'}, inplace=True)
    return (average_df)

def make_linear_regression_time(df_train):
    # get required columns and add time difference column
    df_sorted = df_train[['case:concept:name', 'concept:name', 'lifecycle:transition', 'time:timestamp']]
    df_sorted.sort_values(['case:concept:name', 'time:timestamp'], inplace=True)
    df_sorted['time_until_next'] = df_sorted['time:timestamp'].shift(-1) - df_sorted['time:timestamp']

    # add relative index
    label_encoder = LabelEncoder()
    df_sorted['encoded_concept:name'] = label_encoder.fit_transform(df_sorted['concept:name'])
    df_sorted['relative_index'] = df_sorted.groupby('case:concept:name').cumcount() + 1
    # remove worthless values
    df_sorted['time_until_next'] = df_sorted['time_until_next'].mask(
        df_sorted['time_until_next'] < pd.Timedelta("0 days"))

    df_train_regression = df_sorted.copy()
    df_train_regression = df_train_regression.dropna()
    df_train_regression['time_until_next_millisec'] = df_train_regression['time_until_next']/ np.timedelta64(1,'ms')
    df_train_regression['z'] = np.abs(stats.zscore(df_train_regression['time_until_next_millisec']))
    df_train_regression = df_train_regression.drop(df_train_regression[df_train_regression['z'] > 3].index)
    
    x_train = df_train_regression['relative_index']
    y_train = df_train_regression['time_until_next_millisec']
    x_train = x_train.to_frame()
    y_train = y_train.to_frame()
    regr = LinearRegression()
    regr.fit(x_train,y_train)
    return regr

def make_baseline_action(df_train):
    # first we drop the columns not needed for the baseline
    df_train_baseline = df_train[['case:concept:name', 'concept:name', 'time:timestamp']]

    # sort the dataframe first by case and then by time
    df_train_baseline = df_train_baseline.sort_values(by=['case:concept:name', 'time:timestamp'])

    # we encode the data (concept:name turns from string into integer)
    label_encoder = LabelEncoder()
    df_t_bs = df_train_baseline.copy()
    df_t_bs['encoded_concept:name'] = label_encoder.fit_transform(df_t_bs['concept:name'])

    # add a relative index to each action in the case
    df_t_bs['relative_index'] = df_t_bs.groupby('case:concept:name').cumcount() + 1

    df_action_ri = df_t_bs[['encoded_concept:name', 'relative_index']]

    # we want to know the action that is most popular at each relative index
    action_counts = df_action_ri.groupby(['relative_index', 'encoded_concept:name']).size().reset_index(name='count')
    most_popular_actions = action_counts.loc[action_counts.groupby('relative_index')['count'].idxmax()]

    # decode the labels
    most_popular_actions['concept:name'] = label_encoder.inverse_transform(most_popular_actions['encoded_concept:name'])

    # make a simple lookup table to find the most popular action
    lookup_table = most_popular_actions[['relative_index', 'concept:name']]

    return(lookup_table)


def make_naive_prediction(df_train, df_test):
    print('prediction maken')
    action_at_index = make_baseline_action(df_train)
    time_at_action = make_baseline_time(df_train)
    regr = make_linear_regression_time(df_train)

    # sort test df on concept and time and add relative index
    df_test = df_test.sort_values(by=['case:concept:name', 'time:timestamp'])
    df_test['relative_index'] = df_test.groupby('case:concept:name').cumcount() + 1

    # because the merge adds a column with the same name we will rename the colum from lookup_table
    action_at_index.rename(columns={'concept:name': 'naive_prediction_action'}, inplace=True)
    # add one to the relative index because you want to predict the next action not the current action
    action_at_index['relative_index'] = action_at_index['relative_index'] - 1
    df_test = df_test.merge(action_at_index, on='relative_index', how='left')

    # then we will merge on the average time untill the next action based on the current action
    df_test = df_test.merge(time_at_action, on='concept:name', how='left')
    
    # linear regression
    x_test = df_test['relative_index']
    x_test = x_test.to_frame()
    df_test['time_regression'] = regr.predict(x_test)
    
    print(df_test)

    df_test.to_csv('naive_prediction.csv', index=False)
    
def split_code(cleaned_df):
    df_train, df_test = train_test_split(cleaned_df, train_size=0.75, random_state=None, shuffle=False, stratify=None)
    invalid_date = df_test.iloc[0].loc['time:timestamp']
    train_traces = df_train['case:concept:name'].loc[df_train['time:timestamp'] > invalid_date]
    train_traces = train_traces.drop_duplicates()
    for trace in train_traces:
        df_train = df_train.drop(df_train[df_train['case:concept:name'] == trace].index)
    return df_train, df_test

def evaluate_prediction(df_test):
    length = len(df_test)
    i = 0
    count = 0
    df_test_shifted = df_test.shift(-1).copy()
    df_test_shifted['time_until_next'] = df_test_shifted['time:timestamp'].shift(-1) - df_test_shifted['time:timestamp']
    df_test_shifted['time_until_next'] = df_test_shifted['time_until_next'].mask(
        df_test_shifted['time_until_next'] < pd.Timedelta("0 days"))
    print(df_test)
    print(df_test_shifted)
    while i < length:
        if df_test.iloc[i].loc['naive_prediction_action'] == df_test_shifted.iloc[i].loc['concept:name']:
            count = count + 1
        i = i + 1            
    accuracy = count/len(df_test) * 100
    print("Event accuracy: {}%".format(accuracy))

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    file_path = input("Enter the name (same folder) or path of the xes file containing the dataset: ")
    event_df, event_log = xes_to_df(file_path)
    cleaned_df = event_df.drop_duplicates()

    df_train, df_test = split_code(cleaned_df)
    make_naive_prediction(df_train, df_test)

