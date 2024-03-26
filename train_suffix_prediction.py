
import pandas as pd
import numpy as np
import pm4py as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras. models import Sequential
from keras.layers import LSTM, Dense, Embedding
import pickle

def create_sequences(df):
        grouped = df.groupby('case:concept:name')
        X, Y = [], []
        for name, group in grouped:
            events = list(group['concept:name'])
            for i in range(1, len(events)):
                X.append(events[:i])
                Y.append(events[i:])
        return X, Y

if __name__ == "__main__":
    # Load the dataset
    file_path = "train.xes"  # Replace with your actual file path
    train_df = pm.read_xes(file_path)

    # Create sequences for training and testing data
    X_train, Y_train = create_sequences(train_df)

    # Tokenize the events
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train + Y_train)
    vocab_size = len(tokenizer.word_index) + 1

    # Convert and pad sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train) 
    max_seq_length = max([len(seq) for seq in X_train_seq])
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length, padding='pre')

    # Prepare output sequences
    Y_train_seq = tokenizer.texts_to_sequences(Y_train)
    Y_train_pad = pad_sequences(Y_train_seq, maxlen=max_seq_length, padding='post')
    Y_train_cat = np.array([to_categorical(seq, num_classes=vocab_size) for seq in Y_train_pad])
    
    # Build the LSTM model
    model = Sequential()
    model.add(Embedding(vocab_size, output_dim=50))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dense(vocab_size, activation='softmax'))

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_pad, Y_train_cat, epochs=1, batch_size=32)
    
    # save model to file
    model.save("suffix_pred_model.keras")
    
    # save encoder to file
    tokenizer_file = "tokenizer.pkl"  
    with open(tokenizer_file, 'wb') as file:
        pickle.dump(tokenizer, file)

    