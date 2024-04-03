
import pandas as pd
import numpy as np
import pm4py as pm
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras. models import Sequential
from keras.layers import LSTM, Dense, Embedding
import pickle
from collections import defaultdict

max_sequence_len = 10  # Adjust as necessary

# Transform dataframe into sequences of events for each trace
def get_event_sequences(df):
    sequences = defaultdict(list)
    for _, row in df.iterrows():
        trace_id = row['case:concept:name']
        event = row['concept:name']
        sequences[trace_id].append(event)
    return sequences

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

if __name__ == "__main__":
    # Load the dataset
    file_path = "train.xes"  # Replace with your actual file path
    train_df = pm.read_xes(file_path)

    # Transforming both training and testing data
    train_sequences = get_event_sequences(train_df)
    
    # Tokenization and data preparation
    tokenizer = Tokenizer(filters='', lower=False, split='\n')
    tokenizer.fit_on_texts(['\n'.join(seq) for seq in train_sequences.values()])
    
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
    model.fit(X_train, y_train, epochs=1, batch_size=128)  # Adjust epochs and batch size as needed

    # aSve model to file
    model.save("suffix_pred_model.keras")
        
    # save encoder to file
    tokenizer_file = "tokenizer.pkl"  
    with open(tokenizer_file, 'wb') as file:
        pickle.dump(tokenizer, file)

    