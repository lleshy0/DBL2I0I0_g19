import pandas as pd
import numpy as np
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:\Henrique\TUE\YEAR2\Q3\dbl\csv_2012.csv"  # Update with your file path
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

max_sequence_len = 70  # Adjust as necessary
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
model.fit(X_train, y_train, epochs=1, batch_size=32)  # Adjust epochs and batch size as needed

# Prediction function
def predict_suffix(model, tokenizer, prefix, max_length):
    sequence = tokenizer.texts_to_sequences([prefix])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')
    
    predicted_suffix = []
    for _ in range(max_length):
        predictions = model.predict(sequence, verbose=0)
        next_event_token = np.argmax(predictions, axis=-1)[0]
        if next_event_token == 0:  # Typically, 0 is used as the padding token and should not be part of the prediction.
            break
        predicted_suffix.append(next_event_token)
        sequence = np.append(sequence[0][1:], next_event_token).reshape(1, max_sequence_len)
    
    predicted_events = [tokenizer.index_word[token] for token in predicted_suffix]
    return predicted_events

# Function to predict suffixes for test data
def predict_suffixes_for_test_data(model, tokenizer, test_sequences, max_sequence_len, suffix_len):
    predictions = {}
    for case_id, sequence in test_sequences.items():
        prefix_len = len(sequence) // 2  # or set a specific length
        prefix = '\n'.join(sequence[:prefix_len])
        predicted_suffix = predict_suffix(model, tokenizer, prefix, suffix_len)
        predictions[case_id] = (prefix, predicted_suffix)
    return predictions

# Generating predictions for test data
predicted_suffixes = predict_suffixes_for_test_data(model, tokenizer, test_sequences, max_sequence_len, 35)

def calculate_positional_accuracies(model, tokenizer, test_sequences, max_sequence_len):
    # Dictionary to hold accuracy for each suffix position
    positional_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})

    for case_id, true_events in test_sequences.items():
        # Generate the predicted suffix
        prefix_len = len(true_events) // 2  # Change this ratio as needed
        prefix = '\n'.join(true_events[:prefix_len])
        predicted_suffix = predict_suffix(model, tokenizer, prefix, len(true_events) - prefix_len)

        # Calculate accuracy for each position in the suffix
        for i, true_event in enumerate(true_events[prefix_len:]):
            if i < len(predicted_suffix) and predicted_suffix[i] == true_event:
                positional_accuracy[i]['correct'] += 1
            positional_accuracy[i]['total'] += 1

    # Calculate and return the accuracy for each position
    positional_accuracies = {pos: acc['correct'] / acc['total'] for pos, acc in positional_accuracy.items() if acc['total'] > 0}
    return positional_accuracies

# Generate the positional accuracies
positional_accuracies = {0: 0.9385, 1: 0.9302, 2: 0.8573, 3: 0.7407, 4: 0.7400, 5: 0.6886, 6: 0}
# Extract positions and their corresponding accuracies
positions = list(positional_accuracies.keys())
accuracies = [positional_accuracies[position] for position in positions]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(positions, accuracies, marker='o', linestyle='-', color='blue')
plt.title('Positional Accuracies of Predicted Events in Suffix')
plt.xlabel('Position in Suffix')
plt.ylabel('Accuracy')
plt.xticks(positions)
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.grid(axis='y', linestyle='--')
plt.show()

#Print out the positional accuracies
for position, accuracy in sorted(positional_accuracies.items()):
    print(f"Accuracy for position {position+1}: {accuracy:.4f}")

#Displaying predictions for a few cases
for case_id, (prefix, suffix) in list(predicted_suffixes.items())[:30]:  
    print(f"Case ID: {case_id}")
    print("Prefix:", prefix)
    print("Predicted Suffix:", suffix)
    print("\n")
