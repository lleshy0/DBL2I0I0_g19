def clean_data():
    data_frame.drop_duplicates()
    
def split_data(event_log):
    o
    
def appendNextAction(data_frame):
     # sort the dataframe first by case and then by time
    data_frame = data_frame.sort_values(by=['case:concept:name', 'time:timestamp'])

    # add a relative index to each action in the case
    data_frame['relative_index'] = data_frame.groupby('case:concept:name').cumcount() + 1
    
    #add a column with the next action relative to the current action
    data_frame['next_action'] = data_frame['case:concept:name'].shift(-1)
    
def train_random_forest(train_features, train_labels):
    # import random forest
    from sklearn.ensemble import RandomForestRegressor
    # instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # train the model on training data
    rf.fit(train_features, train_labels)

if __name__ == "__main__":
    file_path = input("Enter the name (same folder) or path of the xes file containing the dataset: ")
    event_df, event_log = xes_to_df(file_path)
    