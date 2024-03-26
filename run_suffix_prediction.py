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
import train_suffix_prediction as sp

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
    # Load the dataset
    file_path = "test.xes"  # Replace with your actual file path
    test_df = pm.read_xes(file_path)
    
    # Create sequences for test data
    X_test, Y_test = sp.create_sequences(test_df)
    
    # Import tokenizer
    tokenizer_file = 'tokenizer.pkl'
    with open(tokenizer_file, 'rb') as file:  
        tokenizer = pickle.load(file)
    vocab_size = len(tokenizer.word_index) + 1
        
    # Convert and pad sequences
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    max_seq_length = max([len(seq) for seq in X_test_seq]) #174
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length, padding='pre')
    
    # Prepare output sequences
    Y_test_seq = tokenizer.texts_to_sequences(Y_test)
    Y_test_pad = pad_sequences(Y_test_seq, maxlen=max_seq_length, padding='post')
    Y_test_cat = np.array([to_categorical(seq, num_classes=vocab_size) for seq in Y_test_pad])
    
    # Import model
    model = keras.models.load_model("suffix_pred_model.keras")
    
    # Prediction and Evaluation
    mae, mse = evaluate_model(model, X_test_pad, Y_test_cat, tokenizer)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    