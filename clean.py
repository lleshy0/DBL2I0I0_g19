import pm4py
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def xes_to_df(file_path):
    event_log = pm4py.read_xes(file_path)
    event_df = pm4py.convert_to_dataframe(event_log)
    
    return event_df

if __name__ == "__main__":
    file_path = r""
    event_df = xes_to_df(file_path)
    cleaned_df = event_df.drop_duplicates()

    train_traces, test_traces= np.split(cleaned_df, [int(.75 *len(cleaned_df))])

    sorted_test = test_traces.sort_values(by='case:REG_DATE')

    lowest_start_time = sorted_test['case:REG_DATE'].iloc[0]

    train_traces = train_traces[train_traces['time:timestamp'] <= lowest_start_time]

    print(train_traces)