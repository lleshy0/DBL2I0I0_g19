import suffix_prediction as sp

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


def full_prediction(event_df):
    train_df, test_df = split_data(event_df)
    
    # import event prediction model
    
    
    # import time prediction model
    
    # run prediction

if __name__ == "__main__":
    file_path = "BPI_Challenge_2012.xes"
    event_df, event_log = xes_to_df(file_path)
    full_prediction(event_df)