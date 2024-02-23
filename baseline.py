import pm4py as pm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def xes_to_df(file_path):
    event_log = pm.read_xes(file_path)
    event_df = pm.convert_to_dataframe(event_log)
    
    return event_df, event_log

def make_baseline(df_train):
    ##tracefilter_log_pos = pm.filter_event_attribute_values(df_train, "case:concept:name", ["173688"], level="event", retain=True)
    df_train.sort_values('case:concept:name')
    print(df_train)
    df_train['time_until_next'] = df_train["time:timestamp"].shift(-1) - df_train["time:timestamp"]

    submitted_df = df_train.loc[df_train['concept:name'] == "A_SUBMITTED"].copy()

    average_delay = submitted_df['time_until_next'].mean()
    print(average_delay)

    average_df = pd.DataFrame()

    for case in df_train['concept:name']:
        temp_df = df_train.loc[df_train['concept:name'] == case].copy()
        temp_data = [[case, temp_df['time_until_next'].mean()]]
        # print(average_df)
        average_df = average_df._append(temp_data, ignore_index = True)

    print(average_df)

def make_baseline_action(df_train):

    #first we drop the columns not needed for the baseline
    df_train_baseline = df_train[['case:concept:name','concept:name','time:timestamp']]

    #sort the dataframe first by case and then by time
    df_train_baseline = df_train_baseline.sort_values(by=['case:concept:name','time:timestamp'])

    #we encode the data (concept:name turns from string into integer)
    label_encoder = LabelEncoder()
    df_t_bs = df_train_baseline.copy()
    df_t_bs['encoded_concept:name'] = label_encoder.fit_transform(df_t_bs['concept:name'])

    #add a relative index to each action in the case
    df_t_bs['relative_index'] = df_t_bs.groupby('case:concept:name').cumcount() + 1

    df_action_ri = df_t_bs[['encoded_concept:name','relative_index']]

    #we want to know the action that is most popular at each relative index
    action_counts = df_action_ri.groupby(['relative_index','encoded_concept:name']).size().reset_index(name='count')
    most_popular_actions = action_counts.loc[action_counts.groupby('relative_index')['count'].idxmax()]

    #decode the labels
    most_popular_actions['concept:name'] = label_encoder.inverse_transform(most_popular_actions['encoded_concept:name'])

    #make a simple lookup table to find the most popular action
    lookup_table = most_popular_actions[['relative_index','concept:name']]

    print(lookup_table)

                            

if __name__ == "__main__":
    file_path = r"C:\Users\20191663\Documents\Y5\Y5Q3\2IOI0 - DBL process mining\BPI Challenge 2012_1_all\BPI_Challenge_2012.xes\BPI_Challenge_2012.xes"
    event_df, event_log = xes_to_df(file_path)
    cleaned_df = event_df.drop_duplicates()

    df_train, df_test = train_test_split(cleaned_df, train_size=0.75, random_state=None, shuffle=False, stratify=None)
    make_baseline_action(df_train)
