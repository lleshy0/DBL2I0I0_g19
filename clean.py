import pm4py
from sklearn.model_selection import train_test_split

def xes_to_df(file_path):
    event_log = pm4py.read_xes(file_path)
    event_df = pm4py.convert_to_dataframe(event_log)
    
    return event_df

if __name__ == "__main__":
    file_path = r"C:\Users\macie\Downloads\BPI_Challenge_2012.xes.gz"
    event_df = xes_to_df(file_path)
    dropna_df = event_df.dropna()
    cleaned_df = dropna_df.drop_duplicates()
    

    cleaned_df_train = train_test_split(cleaned_df, train_size=0.75, random_state=None, shuffle=True, stratify=None)
    print(cleaned_df_train)
    
