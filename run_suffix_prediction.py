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
from pyxdameraulevenshtein import damerau_levenshtein_distance

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

if __name__ == "__main__":
    # Load the dataset
    file_path = "test.xes"  # Replace with your actual file path
    test_df = pm.read_xes(file_path)
    
    # Import tokenizer
    tokenizer_file = 'tokenizer.pkl'
    with open(tokenizer_file, 'rb') as file:  
        tokenizer = pickle.load(file)
    vocab_size = len(tokenizer.word_index) + 1
        
    # Import model
    model = keras.models.load_model("suffix_pred_model.keras", compile=False)
    max_sequence_len = sp.max_sequence_len
    
    # Prepare data
    test_sequences = sp.get_event_sequences(test_df)
    
    # Generating predictions for test data
    predicted_suffixes = predict_suffixes_for_test_data(model, tokenizer, test_sequences, max_sequence_len, 5)
    
    # Displaying predictions for a few cases
    for case_id, (prefix, suffix) in list(predicted_suffixes.items())[:5]:  # Display first 5 cases
        print(f"Case ID: {case_id}")
        print("Prefix:", prefix)
        print("Predicted Suffix:", suffix)
        print("\n")