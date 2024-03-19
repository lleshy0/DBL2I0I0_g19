import pandas as pd
from dateutil import parser
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras import utils
# Load and preprocess the dataset
def load_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['time:timestamp'] = df['time:timestamp'].apply(lambda x: parser.parse(x))
    df_sorted = df.sort_values('time:timestamp')
    
    split_index = int(len(df_sorted) * 0.75)
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]

    train_cases = set(train_df['case:concept:name'])
    test_cases = set(test_df['case:concept:name'])
    overlapping_cases = train_cases & test_cases

    return (train_df[~train_df['case:concept:name'].isin(overlapping_cases)],
            test_df[~test_df['case:concept:name'].isin(overlapping_cases)])

# Encoding and creating sequences
def encode_activities(train_df, test_df):
    activity_encoder = LabelEncoder()
    train_df['encoded_activity'] = activity_encoder.fit_transform(train_df['concept:name'])
    test_df['encoded_activity'] = activity_encoder.transform(test_df['concept:name'])
    return train_df, test_df, activity_encoder

def create_sequences(df, group_by_case, activity_col, max_sequence_len):
    sequences = []
    for _, group in df.groupby(group_by_case):
        sequence = list(group[activity_col])
        if len(sequence) < max_sequence_len:
            sequence += [0] * (max_sequence_len - len(sequence))
        sequences.append(sequence[:max_sequence_len])
    return np.array(sequences)

def prepare_targets(sequences, activity_encoder):
    return np.array([keras.to_categorical(seq[1:], num_classes=len(activity_encoder.classes_)) for seq in sequences])

# Build the LSTM model
def build_lstm_model(input_shape, output_dim):
    model = keras.Sequential([
        keras.LSTM(50, input_shape=input_shape),
        keras.Dropout(0.2),
        keras.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    file_path = 'path_to_your_file.csv'  # Update this to your file path
    train_df, test_df = load_preprocess_data(file_path)

    train_df, test_df, activity_encoder = encode_activities(train_df, test_df)

    MAX_SEQUENCE_LEN = 20  # Adjust based on your data
    train_sequences = create_sequences(train_df, 'case:concept:name', 'encoded_activity', MAX_SEQUENCE_LEN)
    test_sequences = create_sequences(test_df, 'case:concept:name', 'encoded_activity', MAX_SEQUENCE_LEN)

    train_targets = prepare_targets(train_sequences, activity_encoder)
    test_targets = prepare_targets(test_sequences, activity_encoder)

    model = build_lstm_model((MAX_SEQUENCE_LEN, len(activity_encoder.classes_)), len(activity_encoder.classes_))

    # Train the model
    EPOCHS = 10
    BATCH_SIZE = 64
    model.fit(train_sequences, train_targets, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_sequences, test_targets)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
