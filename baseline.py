import pm4py as pm
import pandas as pd
from sklearn.model_selection import train_test_split

def xes_to_df(file_path):
    event_log = pm.read_xes(file_path)
    event_df = pm.convert_to_dataframe(event_log)
    
    return event_df, event_log

def make_baseline(df_train):
    ##tracefilter_log_pos = pm.filter_event_attribute_values(df_train, "case:concept:name", ["173688"], level="event", retain=True)
    df_train.sort_values('case:concept:name')
    df_train['time_until_next'] = df_train["time:timestamp"].shift(-1) - df_train["time:timestamp"]
        
    submitted_df = df_train.loc[df_train['concept:name'] == "A_SUBMITTED"].copy()  
        
    average_delay = submitted_df['time_until_next'].mean()
    print(average_delay)
    
    average_df = pd.DataFrame()
    
    for case in df_train['concept:name']:
        temp_df = df_train.loc[df_train['concept:name'] == case].copy()
        temp_data = [[case, temp_df['time_until_next'].mean()]]
        print(average_df)
        average_df = average_df._append(temp_data, ignore_index = True)
    
    print(average_df)
                            

if __name__ == "__main__":
    file_path = "BPI_Challenge_2012.xes.gz"
    event_df, event_log = xes_to_df(file_path)
    cleaned_df = event_df.drop_duplicates()

    df_train, df_test = train_test_split(cleaned_df, train_size=0.75, random_state=None, shuffle=False, stratify=None)
    make_baseline(df_train)
