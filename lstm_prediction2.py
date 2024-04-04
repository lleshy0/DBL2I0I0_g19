import pm4py as pm
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
import warnings

def predict(df_test):
    df_test_copy = df_test.copy()
    
    # get required columns and add time difference column
    df_test_copy.sort_values(['case:concept:name', 'time:timestamp'], inplace=True)
    df_test_copy['time_until_next'] = df_test_copy['time:timestamp'].shift(-1) - df_test_copy['time:timestamp']
    df_test_copy['datetime'] = pd.to_datetime(df_test_copy['time:timestamp'])
    df_test_copy['time_until_next_millisec'] = df_test_copy['time_until_next']/ np.timedelta64(1,'ms')
    
    df_test_copy['weekday'] = df_test_copy['time:timestamp'].dt.dayofweek
    df_test_copy['hour'] = pd.to_datetime(df_test_copy['time:timestamp'], format='%Y-%m-%d %H:%M:%S')
    df_test_copy['hour'] = df_test_copy['hour'].dt.hour
    
    df_test_lstm = df_test_copy[['time_until_next_millisec','weekday','hour']]
    
    scalar = StandardScaler()
    scalar = scalar.fit(df_test_lstm)
    df_test_lstm_scaled = scalar.transform(df_test_lstm)
    
    x_test, y_test = df_to_X_y(df_test_lstm_scaled, df_test_lstm)
    
    model_lstm = load_model('model_lstm2/')
    
    prediction = model_lstm.predict(-x_test[5:])
    prediction_copies = np.repeat(prediction, 3, axis=1)
    real_pred = scalar.inverse_transform(prediction_copies)[:,0]
    
    
    real_pred = np.insert(real_pred, 0, np.repeat(np.nan, 10))
    print(real_pred.shape)
    print(df_test_copy.shape)
    
    df_test_copy['lstm_prediction'] = real_pred.tolist()
    df_test_copy['lstm_prediction'] = df_test_copy['lstm_prediction'] * -1
    df_test_copy['time_prediction'] = pd.to_timedelta(df_test_copy['lstm_prediction'], unit='ms')
    df_test_copy['predicted_time'] = df_test_copy['time:timestamp'].shift(1) + df_test_copy['time_prediction'].shift(1)
    df_test_copy.to_csv('time_lstm2.csv', index=False)
    print(df_test_copy)
    evaluate(df_test_copy)
    
def evaluate(df_test):
    df = df_test.copy()
    print(df.shape)
    df = df.dropna()
    print(df.shape)
    mea = mean_absolute_error(df['time_until_next_millisec'], df['lstm_prediction'])
    mea = pd.to_timedelta(mea, unit='ms')
    print(mea)
    
    
def df_to_X_y(df_scaled,df_raw,fut=1,past=5):
    x = []
    y = []
    for i in range(past, len(df_scaled)- fut + 1):
        x.append(df_scaled[i - past:i , 0:df_raw.shape[1]])
        y.append(df_scaled[i + fut - 1:i + fut, 0])
    x = np.asarray(x).astype('float32')    
    return np.array(x), np.array(y)  

def split_data(cleaned_df):

    train_traces, test_traces= np.split(cleaned_df, [int(.75 *len(cleaned_df))])

    sorted_test = test_traces.sort_values(by='case:REG_DATE')

    lowest_start_time = sorted_test['case:REG_DATE'].iloc[0]

    train_traces = train_traces[train_traces['time:timestamp'] <= lowest_start_time]

    return train_traces, test_traces

def xes_to_df(file_path):
    event_log = pm.read_xes(file_path)
    event_df = pm.convert_to_dataframe(event_log)

    return event_df, event_log

if __name__ == "__main__":
    file_path = "BPI_Challenge_2012.xes.gz"
    warnings.filterwarnings('ignore')
    event_df, event_log = xes_to_df(file_path)
    # split data into train and test sets
    train_df, test_df = split_data(event_df)
    predict(test_df)