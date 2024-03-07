import pm4py as pm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pm4py.objects.conversion.log import converter as xes_converter
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings

def xes_to_df(file_path):
    event_log = pm.read_xes(file_path)
    event_df = pm.convert_to_dataframe(event_log)

    return event_df, event_log

def get_next_event(group):
    group['next_event'] = group['concept:name'].shift(-1)

    return group

def append_next_event(df):
     # sort the dataframe first by case and then by time
    df = df.sort_values(by=['case:concept:name', 'time:timestamp'])

    # add a column with the next action relative to the current action
    df = df.groupby('case:concept:name', group_keys=False).apply(get_next_event)

    # the last event in every trace will get a NaN value. We fill it with a custom marker
    df['next_event'] = df['next_event'].fillna('[TRACE_END]')
    
    return df

def split_data(cleaned_df):

    train_traces, test_traces= np.split(cleaned_df, [int(.75 *len(cleaned_df))])

    sorted_test = test_traces.sort_values(by='case:REG_DATE')

    lowest_start_time = sorted_test['case:REG_DATE'].iloc[0]

    train_traces = train_traces[train_traces['time:timestamp'] <= lowest_start_time]

    return train_traces, test_traces

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

def make_naive_prediction(df_train, df_test):
    
    action_at_index = make_baseline_action(df_train)
    time_at_action = make_baseline_time(df_train)
    regr = make_linear_regression_time(df_train)
    temp = pd.DataFrame()

    # sort test df on concept and time and add relative index
    temp = df_test.sort_values(by=['case:concept:name', 'time:timestamp'])
    temp['relative_index'] = temp.groupby('case:concept:name').cumcount() + 1

    # because the merge adds a column with the same name we will rename the colum from lookup_table
    action_at_index.rename(columns={'concept:name': 'naive_event'}, inplace=True)
    # add one to the relative index because you want to predict the next action not the current action
    action_at_index['relative_index'] = action_at_index['relative_index'] - 1
    temp = temp.merge(action_at_index, on='relative_index', how='left')

    # then we will merge on the average time untill the next action based on the current action
    temp = temp.merge(time_at_action, on='concept:name', how='left')
    
    # linear regression
    x_test = temp['relative_index']
    x_test = x_test.to_frame()
    temp['time_regression'] = regr.predict(x_test)
    
    # rename for consistency
    temp.rename(columns={'time_until_next': 'naive_time'}, inplace=True)
    
    return temp[['naive_event', 'naive_time', 'time_regression']]

def predict_time(train_df, test_df):
    
    regr = make_linear_regression_time(train_df)
    
    # to be added
    # regression time = ......
    
    return regression_time
    
def split_timestamps(df):
    temp = pd.DataFrame()
    temp['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    # Extract components
    temp['year'] = temp['time:timestamp'].dt.year
    temp['month'] = temp['time:timestamp'].dt.month
    temp['day'] = temp['time:timestamp'].dt.day
    temp['hour'] = temp['time:timestamp'].dt.hour
    temp['minute'] = temp['time:timestamp'].dt.minute

    temp.drop(['time:timestamp'], axis=1, inplace=True)
    
    return temp
    
def prep_features(df, enc1, enc2):

    # create new df without the 'next_event' column
    features = df.drop(['next_event'], axis=1)

    # add year, month, day, hour, minute columns for each timestamp
    features = pd.concat([features, split_timestamps(features)], axis=1)

    # drop timestamp columns
    features.drop(['time:timestamp'], axis=1, inplace=True)
    features.drop(['case:REG_DATE'], axis=1, inplace=True)

    # encode categorical variables
    features['concept:name'] = enc1.fit_transform(features['concept:name'])
    features['lifecycle:transition'] = enc2.fit_transform(features['lifecycle:transition'])
    
    # return df with added encoding
    return features
    
def train_random_forest(train_features, train_labels):
    # instantiate model with n decision trees
    rf = RandomForestClassifier(n_estimators = 1, random_state = 42)

    # train the model on training data
    rf.fit(train_features, train_labels)
    
    return rf

def predict_next_event(train_df, test_df):
    # label encoder for concept:name
    enc1 = LabelEncoder()
    
    # label encoder for lifecycle:transition
    enc2 = LabelEncoder()
    
    # encode train labels and features
    train_labels = enc1.fit_transform(train_df['next_event'])
    train_features = prep_features(train_df, enc1, enc2)
    
    # encode test labels and features
    test_labels = enc1.fit_transform(test_df['next_event'])
    test_features = prep_features(test_df, enc1, enc2)
    
    # train event prediction model
    rf = train_random_forest(train_features, train_labels)
    
    # perform prediction on test features
    predicted_labels = rf.predict(test_features)
    
    # print prediction results
    print('Accuracy score of event prediction using random forest: ')
    print(evaluate_event_prediction(test_labels, predicted_labels))
    
    # remove added columns from test_df
    test_df.drop(['next_event'], axis=1, inplace=True)
    
    # format results to a dataframe
    temp = pd.DataFrame()
    temp['rand_forest_event'] = enc1.inverse_transform(predicted_labels)
    
    return temp

def evaluate_event_prediction(test_labels, prediction_labels):
    # set up temp df
    df = pd.DataFrame()
    df['A'] = test_labels
    df['B'] = prediction_labels
    
    # Count the number of rows where test equals prediction
    same_values_count = len(df[df['A'] == df['B']])

    # Total number of rows in the DataFrame
    total_rows = len(df)

    # Calculate the percentage
    return (same_values_count / total_rows) * 100

def full_prediction(event_df):
    # drop duplicate rows
    event_df.drop_duplicates()
    
    # add a 'next_event' column for event prediction
    event_df = append_next_event(event_df)
    
    # split data into train and test sets
    train_df, test_df = split_data(event_df)
    
    # obtain column for predicted event
    event_prediction = predict_next_event(train_df, test_df)
    
    # clean up 'next_event' column after event prediction
    train_df.drop(['next_event'], axis = 1, inplace=True)
    
    # rest of prediction
    prediction = make_naive_prediction(train_df, test_df)
    
    # concatenate all prediction columns to the original test set
    prediction = pd.concat([test_df, prediction, event_prediction])
    
    # print and export file
    print(prediction)
    prediction.to_csv('tool2.csv', index=False)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    file_path = input("Enter the path of the xes file containing the dataset: ")
    event_df, event_log = xes_to_df(file_path)
    full_prediction(event_df)