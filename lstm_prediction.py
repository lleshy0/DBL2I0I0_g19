import pm4py as pm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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
    
    x_test, y_test = df_to_X_y(df_test_copy[['time_until_next_millisec']], 5)
    
    df_test_copy = df_test_copy.iloc[5:]
    
    model_lstm = load_model('lstm_time_model.keras')
    
    df_test_copy['lstm_prediction'] = model_lstm.predict(x_test).flatten() 
    df_test_copy['time_prediction'] = pd.to_timedelta(df_test_copy['lstm_prediction'], unit='ms')
    evaluate(df_test_copy)
    
def evaluate(df_test):
    df = df_test.copy()
    df = df.dropna()
    mea = mean_absolute_error(df['time_until_next_millisec'], df['lstm_prediction'])
    mea = pd.to_timedelta(mea, unit='ms')
    print("Time prediction - mean absolute error: ")
    print(mea)
    
    
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

if __name__ == "__main__":
    file_path = "test.xes"
    warnings.filterwarnings('ignore')
    test_df = pm.read_xes(file_path)
    predict(test_df)