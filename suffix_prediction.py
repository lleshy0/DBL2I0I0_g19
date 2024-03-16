import pm4py as pm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from pm4py.objects.conversion.log import converter as xes_converter
import warnings
import pickle

def xes_to_df(file_path):
    event_log = pm.read_xes(file_path)
    event_df = pm.convert_to_dataframe(event_log)

    return event_df, event_log

def get_next_event(group):
    group['next_event'] = group['concept:name'].shift(-1)

    return group

def append_next_event(df):
    # sort the dataframe first by case and then by time
    temp = df.sort_values(by=['case:concept:name', 'time:timestamp']).copy()

    # add a column with the next action relative to the current action
    temp = temp.groupby('case:concept:name', group_keys=False).apply(get_next_event)

    # the last event in every trace will get a NaN value. We fill it with a custom marker
    temp['next_event'] = temp['next_event'].fillna('[TRACE_END]')
    
    return temp['next_event']

def get_previous_event(group):
    group['previous_event'] = group['concept:name'].shift()
    
    return group

def append_previous_event(df):
    # sort the dataframe first by case and then by time
    temp = df.sort_values(by=['case:concept:name', 'time:timestamp']).copy()

    # add a column with the next action relative to the current action
    temp = temp.groupby('case:concept:name', group_keys=False).apply(get_previous_event)
    
     # the first event in every trace will get a NaN value. We fill it with a custom marker
    temp['previous_event'] = temp['previous_event'].fillna('[TRACE_START]')
    
    return temp['previous_event']

def split_timestamps(df):
    temp = pd.DataFrame()
    temp['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    # Extract components
    temp['year'] = temp['time:timestamp'].dt.year
    temp['month'] = temp['time:timestamp'].dt.month
    temp['day'] = temp['time:timestamp'].dt.day
    temp['hour'] = temp['time:timestamp'].dt.hour
    temp['weekday'] = temp['time:timestamp'].dt.weekday

    temp.drop(['time:timestamp'], axis=1, inplace=True)
    
    return temp

def prep_features(df):
    # make a copy of the og dataframe
    features = df.copy()
    
    # add previous event
    temp = pd.DataFrame()
    temp['previous_event'] = append_previous_event(features)
    temp = pd.get_dummies(temp['previous_event'], dtype=int)
    features = pd.concat([features, temp], axis=1)
    
    # add year, month, day, hour, minute columns for each timestamp
    features = pd.concat([features, split_timestamps(features)], axis=1)

    # drop timestamp columns
    features.drop(['time:timestamp'], axis=1, inplace=True)
    features.drop(['case:REG_DATE'], axis=1, inplace=True)

    # encode categorical variables
    features = pd.concat([features, pd.get_dummies(features['concept:name'], dtype=int)], axis=1)
    
    # drop reduandant/unused columns
    features.drop(['concept:name'], axis=1, inplace=True)
    features.drop(['lifecycle:transition'], axis=1, inplace=True)
    features.drop(['case:AMOUNT_REQ'], axis=1, inplace=True)
    
    # return df with added encoding
    return features

def train_model(train_df):
    features = prep_features(train_df)
    labels = append_next_event(train_df)
    
    # instantiate model with n decision trees
    rf = RandomForestClassifier(n_estimators = 100, random_state = 42)

    # train the model on training data
    rf.fit(train_features, train_labels)
    
    return rf

if __name__ == "__main__":
    file_path = "BPI_Challenge_2012.xes"
    train_df, event_log = xes_to_df(file_path)
    #print(train_df)
    print(prep_features(train_df))