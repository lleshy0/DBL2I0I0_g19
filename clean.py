import pm4py

def xes_to_df(file_path):
    event_log = pm4py.read_xes(file_path)
    event_df = pm4py.convert_to_dataframe(event_log)
    
    return event_df

if __name__ == "__main__":
    file_path = r"C:\Users\20221393\OneDrive - TU Eindhoven\Desktop\BPI Challenge 2012_1_all\BPI_Challenge_2012.xes\BPI_Challenge_2012.xes"
    event_df = xes_to_df(file_path)
    dropna_df = event_df.dropna()
    cleaned_df = dropna_df.drop_duplicates()
    print(cleaned_df)
    
    
