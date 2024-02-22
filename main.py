import pm4py

def import_xes(file_path):
    event_log = pm4py.read_xes(file_path)
    event_df = pm4py.convert_to_dataframe(event_log)
    print(event_df)
    start_activities = pm4py.get_start_activities(event_log)
    end_activities = pm4py.get_end_activities(event_log)
    print("Start activities: {}\nEnd activities: {}".format(start_activities, end_activities))

if __name__ == "__main__":
    file_path = r"C:\Users\20191663\Documents\Y5\Y5Q3\2IOI0 - DBL process mining\BPI Challenge 2012_1_all\BPI_Challenge_2012.xes\BPI_Challenge_2012.xes"
    import_xes(file_path)
