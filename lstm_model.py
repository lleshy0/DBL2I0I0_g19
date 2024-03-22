import pm4py as pm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import datetime
from pm4py.objects.conversion.log import converter as xes_converter
from sklearn.linear_model import LinearRegression
from scipy import stats
import pickle
import warnings
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
import tensorflow as tf

def lstm_time(df_train,df_test):
    # get required columns and add time difference column
    df_sorted = df_train[['case:concept:name', 'concept:name', 'lifecycle:transition', 'time:timestamp']]
    df_sorted.sort_values(['case:concept:name', 'time:timestamp'], inplace=True)
    df_sorted['time_until_next'] = df_sorted['time:timestamp'].shift(-1) - df_sorted['time:timestamp']
    df_sorted['datetime'] = pd.to_datetime(df_sorted['time:timestamp'])

    # add relative index
    label_encoder = LabelEncoder()
    df_sorted['encoded_concept:name'] = label_encoder.fit_transform(df_sorted['concept:name'])
    df_sorted['relative_index'] = df_sorted.groupby('case:concept:name').cumcount() + 1
    # remove worthless values
    df_sorted['time_until_next'] =df_sorted['time_until_next'].mask(
        df_sorted['time_until_next'] < pd.Timedelta("0 days"), other = pd.Timedelta("0 days"))
    
    df_train_regression = df_sorted.copy()
    df_train_regression = df_train_regression.dropna()
    df_train_regression['time_until_next_millisec'] = df_train_regression['time_until_next']/ np.timedelta64(1,'ms')
    df_train_regression['z'] = np.abs(stats.zscore(df_train_regression['time_until_next_millisec']))
    df_train_regression = df_train_regression.drop(df_train_regression[df_train_regression['z'] > 3].index)
    
    x_array, y_array = df_to_X_y(df_train_regression[['time_until_next_millisec']], 5)
    x_array_train, y_array_train = x_array[:int(0.8 * len(x_array))], y_array[:int(0.8 * len(y_array))] 
    x_array_valid, y_array_valid = x_array[int(0.8 * len(x_array)):], y_array[int(0.8 * len(y_array)):]
    
    model_lstm = Sequential()
    model_lstm.add(InputLayer((5,1)))
    model_lstm.add(LSTM(64))
    model_lstm.add(Dense(8,'relu'))
    model_lstm.add(Dense(1,'linear'))
    
    mcp = ModelCheckpoint('model_lstm/', save_best_only = True)
    model_lstm.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.1), metrics=[RootMeanSquaredError()])
    
    model_lstm.fit(x_array_train,y_array_train,validation_data=(x_array_valid,y_array_valid),epochs=10, callbacks=[mcp])
        

def df_to_X_y(df,window_size=5):
    df_np = df.to_numpy()
    x = []
    y = []
    for i in range(len(df_np) - window_size):
        row = [[a] for a in df_np[i:i + window_size]]
        x.append(row)
        label = df_np[i + window_size]
        y.append(label)
    x = np.asarray(x).astype('float32')    
    return np.array(x), np.array(y)

def xes_to_df(file_path):
    event_log = pm.read_xes(file_path)
    event_df = pm.convert_to_dataframe(event_log)

    return event_df, event_log

def split_data(cleaned_df):

    train_traces, test_traces= np.split(cleaned_df, [int(.75 *len(cleaned_df))])

    sorted_test = test_traces.sort_values(by='case:REG_DATE')

    lowest_start_time = sorted_test['case:REG_DATE'].iloc[0]

    train_traces = train_traces[train_traces['time:timestamp'] <= lowest_start_time]

    return train_traces, test_traces

def clean_data(data_frame):
    return data_frame.drop_duplicates()



if __name__ == "__main__":
    file_path = "BPI_Challenge_2012.xes.gz"
    warnings.filterwarnings('ignore')
    event_df, event_log = xes_to_df(file_path)
    event_df = clean_data(event_df)
    # drop duplicate rows
    event_df.drop_duplicates()
    # split data into train and test sets
    train_df, test_df = split_data(event_df)
    lstm_time(train_df, test_df)