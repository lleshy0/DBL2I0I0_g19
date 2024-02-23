import pm4py as pm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def xes_to_df(file_path):
    event_log = pm.read_xes(file_path)
    event_df = pm.convert_to_dataframe(event_log)
    
    return event_df, event_log

def make_baseline_time(df_train):
    #get required columns and add time difference column
    df_sorted = df_train[['case:concept:name','concept:name','lifecycle:transition','time:timestamp']]
    df_sorted.sort_values(['case:concept:name', 'time:timestamp'], inplace = True)
    df_sorted['time_until_next'] = df_sorted['time:timestamp'].shift(-1) - df_sorted['time:timestamp']
    
    #add relative index
    label_encoder = LabelEncoder()
    df_sorted['encoded_concept:name'] = label_encoder.fit_transform(df_sorted['concept:name'])
    df_sorted['relative_index'] = df_sorted.groupby('case:concept:name').cumcount() + 1
    #remove worthless values
    df_sorted['time_until_next'] = df_sorted['time_until_next'].mask(df_sorted['time_until_next'] < pd.Timedelta("0 days"))
    
    #tracefilter_log_pos = pm.filter_event_attribute_values(df_sorted, "case:concept:name", ["173688"], level="event", retain=True)

    #setup the dataframe
    average_df = pd.DataFrame()
    cases = ['A_SUBMITTED', 'A_PARTLYSUBMITTED', 'A_PREACCEPTED', 'W_Completeren aanvraag', 'A_ACCEPTED', 'O_SELECTED',  'A_FINALIZED', 
             'O_CREATED', 'O_SENT', 'O_SENT_BACK' ,'W_Valideren aanvraag',  'A_REGISTERED', 'A_APPROVED', 'O_ACCEPTED', 
             'W_Wijzigen contractgegevens', 'A_ACTIVATED', 'A_DECLINED', 'A_CANCELLED','W_Completeren aanvraag', 
             'W_Nabellen incomplete dossiers', 'W_Afhandelen leads', 'W_Nabellen offertes', 'W_Beoordelen fraude']

    #get all the average times for the tasks
    for case in cases:
        temp_df = df_sorted.loc[df_sorted['concept:name'] == case].copy()
        temp_data = [[case, temp_df['time_until_next'].mean()]]
        average_df = average_df._append(temp_data, ignore_index = True)

    print(average_df)
    return(average_df)

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
    file_path = "BPI_Challenge_2012.xes.gz"
    event_df, event_log = xes_to_df(file_path)
    cleaned_df = event_df.drop_duplicates()

    df_train, df_test = train_test_split(cleaned_df, train_size=0.75, random_state=None, shuffle=False, stratify=None)
    make_baseline_time(df_train)
    make_baseline_action(df_train)
    
    
