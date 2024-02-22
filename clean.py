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
    dropna_df = event_df.dropna()
    cleaned_df = dropna_df.drop_duplicates()
    print(cleaned_df)
    # cleaned_df['trace_id'] = cleaned_df['org:resource'].astype(str) + '_' + cleaned_df['lifecycle:transition'].astype(str)

    # train_traces, test_traces = train_test_split(cleaned_df['trace_id'].unique(), test_size=0.25, random_state=42)


    # train_set = cleaned_df[cleaned_df['trace_id'].isin(train_traces)]
    # test_set = cleaned_df[cleaned_df['trace_id'].isin(test_traces)]


    # train_set = train_set.drop(columns=['trace_id'])
    # test_set = test_set.drop(columns=['trace_id'])


    # train_set.to_csv('train_set.csv', index=False)
    # test_set.to_csv('test_set.csv', index=False)

    
