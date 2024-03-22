
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras. models import Sequential
from keras.layers import LSTM, Dense, Embedding

# Load the dataset
file_path = "C:\Henrique\TUE\YEAR2\Q3\dbl\csv_2012.csv"  # Replace with your actual file path
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

def create_sequences(df):
        grouped = df.groupby('case:concept:name')
        X, Y = [], []
        for name, group in grouped:
            events = list(group['concept:name'])
            for i in range(1, len(events)):
                X.append(events[:i])
                Y.append(events[i:])
        return X, Y

def evaluate_model(model, X, Y, tokenizer):
        # Predict the sequences
        predictions = model.predict(X)

        # Convert predictions to sequence of integers
        predictions_seq = np.argmax(predictions, axis=-1)

        # Convert true values to sequence of integers
        Y_seq = np.argmax(Y, axis=-1)

        # Calculate MAE and MSE
        mae = mean_absolute_error(Y_seq.flatten(), predictions_seq.flatten())
        mse = mean_squared_error(Y_seq.flatten(), predictions_seq.flatten())

        return mae, mse

if __name__ == "__main__":
    # Function to create sequences and their corresponding outputs
    

    # Create sequences for training and testing data
    X_train, Y_train = create_sequences(train_df_cleaned)
    X_test, Y_test = create_sequences(test_df_cleaned)

    # Tokenize the events
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train + Y_train + X_test + Y_test)
    vocab_size = len(tokenizer.word_index) + 1

    # Convert and pad sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    max_seq_length = max([len(seq) for seq in X_train_seq + X_test_seq])
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length, padding='pre')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length, padding='pre')

    # Prepare output sequences
    Y_train_seq = tokenizer.texts_to_sequences(Y_train)
    Y_test_seq = tokenizer.texts_to_sequences(Y_test)
    Y_train_pad = pad_sequences(Y_train_seq, maxlen=max_seq_length, padding='post')
    Y_test_pad = pad_sequences(Y_test_seq, maxlen=max_seq_length, padding='post')
    Y_train_cat = np.array([to_categorical(seq, num_classes=vocab_size) for seq in Y_train_pad])
    Y_test_cat = np.array([to_categorical(seq, num_classes=vocab_size) for seq in Y_test_pad])

    # Build the LSTM model
    model = Sequential()
    model.add(Embedding(vocab_size, output_dim=50, input_length=max_seq_length))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dense(vocab_size, activation='softmax'))

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_pad, Y_train_cat, epochs=1, batch_size=32)

    # Prediction and Evaluation
    


    mae, mse = evaluate_model(model, X_test_pad, Y_test_cat, tokenizer)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")