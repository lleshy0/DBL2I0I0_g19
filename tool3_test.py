
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "C:\Henrique\TUE\YEAR2\Q3\dbl\csv_2012.csv"  # Replace with your dataset file path
df = pd.read_csv(file_path)

# Preprocessing
df_sorted = df.sort_values('time:timestamp')
split_index = int(len(df_sorted) * 0.75)
train_df = df_sorted.iloc[:split_index]
test_df = df_sorted.iloc[split_index:]
train_cases = set(train_df['case:concept:name'])
test_cases = set(test_df['case:concept:name'])
overlapping_cases = train_cases & test_cases
train_df_cleaned = train_df[~train_df['case:concept:name'].isin(overlapping_cases)]
test_df_cleaned = test_df[~test_df['case:concept:name'].isin(overlapping_cases)]

# Encoding the events and normalizing numerical features
le = LabelEncoder()
train_df_cleaned['encoded_event'] = le.fit_transform(train_df_cleaned['concept:name'])
scaler = StandardScaler()
train_df_cleaned['normalized_amount'] = scaler.fit_transform(train_df_cleaned[['case:AMOUNT_REQ']])

# Preparing the sequences
def create_sequences(df):
    sequences = []
    for name, group in df.groupby('case:concept:name'):
        sequence = group[['encoded_event', 'normalized_amount']].values.tolist()
        sequences.append(sequence)
    return sequences

train_sequences = create_sequences(train_df_cleaned)
max_seq_length = max([len(seq) for seq in train_sequences])
n_features = 2  # encoded_event and normalized_amount

# Padding sequences
padded_sequences = pad_sequences(train_sequences, maxlen=max_seq_length, padding='post', dtype='float32')

# Splitting the dataset into features (X) and target (y)
X = np.array([seq[:-1] for seq in padded_sequences])
y = np.array([seq[1:] for seq in padded_sequences])

# Defining the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(max_seq_length - 1, n_features)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(n_features, activation='linear')))
model.compile(loss='mse', optimizer='adam')

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Prepare test data for prediction
test_df_cleaned['encoded_event'] = le.transform(test_df_cleaned['concept:name'])
test_df_cleaned['normalized_amount'] = scaler.transform(test_df_cleaned[['case:AMOUNT_REQ']])
test_sequences = create_sequences(test_df_cleaned)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_seq_length, padding='post', dtype='float32')
X_test = np.array([seq[:-1] for seq in padded_test_sequences])

# Predicting the suffixes for the test cases
predicted_suffixes = model.predict(X_test)
predicted_suffixes_decoded = [np.argmax(row) for seq in predicted_suffixes for row in seq]

# Extract only the labels (encoded events) for y_test and pad them
y_test_labels = np.array([seq[1:, 0] for seq in padded_test_sequences]) 
y_test_labels_padded = pad_sequences(y_test_labels, maxlen=max_seq_length - 1, padding='post')

# Flatten for comparison with the predictions
y_test_decoded = y_test_labels_padded.flatten()

# Function to calculate accuracy for categorical predictions
def calculate_accuracy(y_true, y_pred):
    correct_predictions = 0
    total_predictions = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_predictions += 1
        total_predictions += 1
    return correct_predictions / total_predictions if total_predictions > 0 else 0

# Calculating metrics
mse = mean_squared_error(y_test_decoded, predicted_suffixes_decoded)
mae = mean_absolute_error(y_test_decoded, predicted_suffixes_decoded)
accuracy = calculate_accuracy(y_test_decoded, predicted_suffixes_decoded)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Accuracy:", accuracy)
