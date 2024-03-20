import pm4py as pm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pm4py.objects.conversion.log import converter as xes_converter
import warnings
import pickle

CATEGORIES = ['A_ACCEPTED', 'A_ACTIVATED', 'A_APPROVED', 'A_CANCELLED', 'A_DECLINED', 'A_FINALIZED', 'A_PARTLYSUBMITTED', 'A_PREACCEPTED', 'A_REGISTERED', 'A_SUBMITTED', 
              'O_ACCEPTED', 'O_CANCELLED', 'O_CREATED', 'O_DECLINED', 'O_SELECTED', 'O_SENT', 'O_SENT_BACK', 'W_Afhandelen leads', 'W_Beoordelen fraude', 'W_Completeren aanvraag', 
              'W_Nabellen incomplete dossiers', 'W_Nabellen offertes', 'W_Valideren aanvraag', 'W_Wijzigen contractgegevens', '[TRACE_START]', '[TRACE_END]']

cat_type = pd.CategoricalDtype(categories=CATEGORIES, ordered=False)

enc_name = LabelEncoder()
enc_name.fit(CATEGORIES)

enc_trans = LabelEncoder()
enc_trans.fit(["START", "SCHEDULE", "COMPLETE"])

def xes_to_df(file_path):
    event_log = pm.read_xes(file_path)
    event_df = pm.convert_to_dataframe(event_log)

    return event_df

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
    #temp['year'] = temp['time:timestamp'].dt.year
    #temp['month'] = temp['time:timestamp'].dt.month
    #temp['day'] = temp['time:timestamp'].dt.day
    temp['hour'] = temp['time:timestamp'].dt.hour
    temp['weekday'] = temp['time:timestamp'].dt.weekday

    temp.drop(['time:timestamp'], axis=1, inplace=True)
    
    return temp

def prep_features(df):
    # make a copy of the og dataframe
    features = df.copy()
    
    # add previous event feature
    temp = pd.DataFrame()
    temp['previous_event'] = enc_name.transform(append_previous_event(features).astype(cat_type))
    features = pd.concat([features, temp], axis=1)
    
    # add relative index feature
    features = features.sort_values(by=['case:concept:name', 'time:timestamp'])
    features['relative_index'] = features.groupby('case:concept:name').cumcount() + 1
    
    # add year, month, day, hour, minute columns for each timestamp
    features = pd.concat([features, split_timestamps(features)], axis=1)

    # drop timestamp columns
    features.drop(['time:timestamp'], axis=1, inplace=True)
    features.drop(['case:REG_DATE'], axis=1, inplace=True)
    
    # encode categorical variable
    features['concept:name'] = enc_name.transform(features['concept:name'].astype(cat_type))

    # drop redundant/unused columns
    features.drop(['case:AMOUNT_REQ'], axis=1, inplace=True)
    features.drop(['case:concept:name'], axis=1, inplace=True)
    
    # encode categorical variable
    features['lifecycle:transition'] = enc_trans.transform(features['lifecycle:transition'])
    
    # return df with added encoding
    return features

def train_model(train_df):
    # prepare features
    features = prep_features(train_df)
    
    # prepare labels
    labels = enc_name.transform(append_next_event(train_df))
    
    # instantiate model with n decision trees
    rf = RandomForestClassifier(n_estimators = 100, random_state = 42)

    # train the model on training data
    rf.fit(features, labels)
    
    return rf

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

if __name__ == "__main__":
    # import train data
    file_path = "train.xes"
    train_df = pm.read_xes(file_path)
    
    # train model
    model = train_model(train_df)
    
    # export model to file
    model_pkl_file = "event_model.pkl"  
    with open(model_pkl_file, 'wb') as file:
        pickle.dump(model, file)
    
    