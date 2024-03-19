import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser
import numpy as np
import pm4py as pm

df = pm.read_xes("BPI_Challenge_2012.xes")

df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

df_sorted = df.sort_values('time:timestamp')

split_index = int(len(df_sorted) * 0.75)
train_df = df_sorted.iloc[:split_index]
test_df = df_sorted.iloc[split_index:]

train_cases = set(train_df['case:concept:name'])
test_cases = set(test_df['case:concept:name'])
overlapping_cases = train_cases & test_cases

train_df_cleaned = train_df[~train_df['case:concept:name'].isin(overlapping_cases)]
test_df_cleaned = test_df[~test_df['case:concept:name'].isin(overlapping_cases)]

pm.write_xes(train_df_cleaned, "train.xes")
pm.write_xes(test_df_cleaned, "test.xes")

#train_df_cleaned.to_csv('train.csv', index=False)
#test_df_cleaned.to_csv('test.csv', index=False)

event_types = df_sorted['concept:name'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(event_types)))
color_dict = dict(zip(event_types, colors))

plt.figure(figsize=(18, 10))

for event_type, color in color_dict.items():
    train_events = train_df_cleaned[train_df_cleaned['concept:name'] == event_type]
    plt.scatter(train_events['time:timestamp'], train_events['case:concept:name'], alpha=0.5, color=color, label=event_type)
    
    test_events = test_df_cleaned[test_df_cleaned['concept:name'] == event_type]
    plt.scatter(test_events['time:timestamp'], test_events['case:concept:name'], alpha=0.5, color=color)

plt.title('Train/Test sets split')
plt.xlabel('Event time:timestamp')
plt.ylabel('Case:concept:name')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

split_timestamp = train_df_cleaned.iloc[-1]['time:timestamp']

plt.axvline(split_timestamp, color='green', linestyle='--', label='Train-Test Split Point')

plt.show()
