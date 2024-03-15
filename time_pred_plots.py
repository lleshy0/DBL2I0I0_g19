import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser
import numpy as np

df = pd.read_csv("naive_prediction.csv")

df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='ISO8601')

df = df.sort_values(by=['case:concept:name', 'time:timestamp'])

df['time_diff'] = df.groupby('case:concept:name')['time:timestamp'].diff()
df['time_until_next'] = pd.to_timedelta(df['time_until_next'])


event_types = df['concept:name'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(event_types)))
color_dict = dict(zip(event_types, colors))

plt.figure(figsize=(18, 10))

# for the normal one colour plot
# plt.scatter(df['time_until_next'], df['time_diff'], alpha=0.2)

# to look at the plot with different colours per event
for event_type, color in color_dict.items():
    df_events = df[df['concept:name'] == event_type]
    plt.scatter(df_events['time_until_next'], df_events['time_diff'], alpha=0.5, color=color,
                label=event_type)
plt.legend()


#making the plot look nice
plt.title('predicted time difference vs actual time difference')
plt.xlabel('predicted time until next')
plt.ylabel('actual time until next')
plt.axline((0, 0), slope=1)

#optional: set axis to the same scale
# plt.gca().set_aspect('equal')

plt.show()
