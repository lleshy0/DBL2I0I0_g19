from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

file_path = "csv_2012.csv"
df = pd.read_csv(file_path)
# Ensure the data is sorted by case ID and then by timestamp to maintain the sequence of events
data_sorted = df.sort_values(by=['case:concept:name', 'time:timestamp'])

# Generate sequences of activities that directly follow each other within each case
traces = data_sorted.groupby('case:concept:name')['concept:name'].apply(list)

# # Concatenate events within each process to create sequences
# sequences = df.groupby('case:concept:name')['concept:name'].apply(tuple).tolist()
#
# # Sort events within each sequence based on their timestamps
# sorted_sequences = []
# for sequence in sequences:
#     # Sort the events based on their timestamps
#     sorted_sequence = sorted(sequence, key=lambda x: df[df['concept:name'] == x].iloc[0]['time:timestamp'])
#     sorted_sequences.append(sorted_sequence)
#     print(sorted_sequence)

# Create pairs of directly following activities (to create the "directly follows" principle)
directly_follows_pairs = traces.apply(lambda x: [(x[i], x[i + 1]) for i in range(len(x) - 1)])

# Flatten the list of directly follows pairs and count the occurrences of each pair
flat_list = [item for sublist in directly_follows_pairs for item in sublist]
pair_counts = Counter(flat_list)

# Extract the 10 most common directly follows pairs
top_10_pairs = pair_counts.most_common(10)

# Prepare data for plotting
pairs_labels = [f'{a} -> {b}' for a, b in [pair[0] for pair in top_10_pairs]]
counts = [pair[1] for pair in top_10_pairs]

# Plotting
plt.figure(figsize=(10, 8))
plt.barh(pairs_labels, counts, color='skyblue')
plt.xlabel('Frequency')
plt.title('Top 10 Most Common Directly Follows Traces')
plt.gca().invert_yaxis() # To display the highest count at the top
plt.show()

data_sorted['time:timestamp'] = pd.to_datetime(data_sorted['time:timestamp'])

# Extract the hour of day from the timestamp for plotting
data_sorted['hour_of_day'] = data_sorted['time:timestamp'].dt.hour

# Use a scatter plot to explore patterns, with different colors for different activities
fig, ax = plt.subplots(figsize=(12, 8))

# Find unique activities and assign colors
activities = data_sorted['concept:name'].unique()
colors = plt.cm.tab20(range(len(activities)))
activity_to_color = dict(zip(activities, colors))

# Plot each activity in its assigned color
for activity, color in activity_to_color.items():
    subset = data_sorted[data_sorted['concept:name'] == activity]
    ax.scatter(subset['hour_of_day'], subset['case:concept:name'], color=color, label=activity, alpha=0.5, edgecolors='none')

ax.grid(True)
ax.legend(title='Activities', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Activity Scatter Plot by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Case ID')
plt.xticks(range(24))

plt.tight_layout()
plt.show()

# Filter the data for events that occur when the bank is closed (between 18:00 and 8:00)
events_during_bank_closed = data_sorted[(data_sorted['time:timestamp'].dt.hour >= 5) | (data_sorted['time:timestamp'].dt.hour < 6)]

# Extract the unique event names during bank closed hours
events_during_bank_closed_names = events_during_bank_closed['concept:name'].unique()

print("Events that occur during bank closed hours (18:00-8:00):")
print(events_during_bank_closed_names)
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

# Identify case:concept:name with a transaction during closed bank hours
closed_bank_hours_cases = df[(df['time:timestamp'].dt.hour >= 18) | (df['time:timestamp'].dt.hour < 8)]['case:concept:name'].unique()

# Select the first 10 case:concept:name with a transaction during closed bank hours
selected_cases = closed_bank_hours_cases[:10]

# Plot transactions for the selected cases during closed bank hours
for selected_case in selected_cases:
    # Filter transactions for the selected case during closed bank hours
    selected_case_transactions = df[df['case:concept:name'] == selected_case]

    # Display the sequence of transactions and timestamps for the selected case
    print(f"Transactions for case:concept:name '{selected_case}' during closed bank hours:")
    for index, row in selected_case_transactions.iterrows():
        print(f"Timestamp: {row['time:timestamp']}, Transaction: {row['concept:name']}")




# Convert 'time:timestamp' to datetime
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='mixed')

# Define working hours and weekdays
start_hour = 9
end_hour = 17
weekdays = [0, 1, 2, 3, 4]  # Monday to Friday

# Create a new column 'during_working_hours' with boolean values
df['during_working_hours'] = ((df['time:timestamp'].dt.hour >= start_hour) &
                              (df['time:timestamp'].dt.hour < end_hour) &
                              (df['time:timestamp'].dt.dayofweek.isin(weekdays)))

# Convert boolean values to string 'True' or 'False'
df['during_working_hours'] = df['during_working_hours'].map({True: 'True', False: 'False'})

# Display the updated DataFrame
print(df)


