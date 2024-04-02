import pandas as pd
import numpy as np
from collections import defaultdict
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Load the dataset
file_path = "c:\Henrique\TUE\YEAR2\Q3\dbl\csv_2012.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Preprocess the dataset
df_sorted = df.sort_values('time:timestamp')
split_index = int(len(df_sorted) * 0.75)
train_df = df_sorted.iloc[:split_index]
test_df = df_sorted.iloc[split_index:]
train_cases = set(train_df['case:concept:name'])
test_cases = set(test_df['case:concept:name'])
overlapping_cases = train_cases & test_cases
train_df_cleaned = train_df[~train_df['case:concept:name'].isin(overlapping_cases)]
test_df_cleaned = test_df[~test_df['case:concept:name'].isin(overlapping_cases)]

# Transform dataframe into sequences of events for each trace
def get_event_sequences(df):
    sequences = defaultdict(list)
    for _, row in df.iterrows():
        trace_id = row['case:concept:name']
        event = row['concept:name']
        sequences[trace_id].append(event)
    return sequences

# Transforming both training and testing data
train_sequences = get_event_sequences(train_df_cleaned)
test_sequences = get_event_sequences(test_df_cleaned)

# Tokenization and data preparation
tokenizer = Tokenizer(filters='', lower=False, split='\n')
tokenizer.fit_on_texts(['\n'.join(seq) for seq in train_sequences.values()])

# Function to create dataset
def create_dataset(sequences, tokenizer, max_sequence_len):
    input_sequences, output_sequences = [], []
    for _, seq in sequences.items():
        token_list = tokenizer.texts_to_sequences(['\n'.join(seq)])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence[:-1])
            output_sequences.append(n_gram_sequence[-1])
    return pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'), np.array(output_sequences)

max_sequence_len = 10  # Adjust as necessary
X_train, y_train = create_dataset(train_sequences, tokenizer, max_sequence_len)
y_train = to_categorical(y_train, num_classes=len(tokenizer.word_index) + 1)

# Building the LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_len))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=5, batch_size=128)  # Adjust epochs and batch size as needed

# Prediction function
def predict_suffix(model, tokenizer, prefix, max_length):
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([prefix])[0]
        sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')
        predicted = model.predict(sequence, verbose=0)
        predicted = np.argmax(predicted, axis=-1)  # Get the class with the highest probability
        if predicted == 0:
            break
        next_event = tokenizer.index_word[predicted[0]]
        prefix += '\n' + next_event
    return prefix.split('\n')[1:]

# Function to predict suffixes for test data
def predict_suffixes_for_test_data(model, tokenizer, test_sequences, max_sequence_len, suffix_len=5):
    predictions = {}
    for case_id, sequence in test_sequences.items():
        prefix_len = len(sequence) // 2  # or set a specific length
        prefix = '\n'.join(sequence[:prefix_len])
        predicted_suffix = predict_suffix(model, tokenizer, prefix, suffix_len)
        predictions[case_id] = (prefix, predicted_suffix)
    return predictions

# Generating predictions for test data
predicted_suffixes = predict_suffixes_for_test_data(model, tokenizer, test_sequences, max_sequence_len, 5)

# Displaying predictions for a few cases
for case_id, (prefix, suffix) in list(predicted_suffixes.items())[:5]:  # Display first 5 cases
    print(f"Case ID: {case_id}")
    print("Prefix:", prefix)
    print("Predicted Suffix:", suffix)
    print("\n")
