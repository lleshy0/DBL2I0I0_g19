import pm4py
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split

def xes_to_df(file_path):
    event_log = pm4py.read_xes(file_path)
    event_df = pm4py.convert_to_dataframe(event_log)
    
    return event_df

if __name__ == "__main__":
    file_path = r"C:\Users\20191663\Documents\Y5\Y5Q3\2IOI0 - DBL process mining\BPI Challenge 2012_1_all\BPI_Challenge_2012.xes\BPI_Challenge_2012.xes"
    event_df = xes_to_df(file_path)
    cleaned_df = event_df.drop_duplicates()

    train_traces, test_traces = pm4py.split_train_test(cleaned_df, train_percentage=0.75, case_id_key='case:concept:name')

    sorted_test = test_traces.sort_values(by='case:REG_DATE')

    lowest_start_time = sorted_test['case:REG_DATE'].iloc[0]

    train_traces = train_traces[train_traces['time:timestamp'] <= lowest_start_time]