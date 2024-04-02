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

def detokenize_suffix(suffix, tokenizer):
    detokenized = []
    index_word = {index: word for word, index in tokenizer.word_index.items()}
    
    for token in suffix:
        detokenized.append(tokenizer.index_word.get(token, None))
    return detokenized

def evaluate_event_suffix_predictions(true_suffixes_list, predicted_suffixes_list):
    if len(true_suffixes_list) != len(predicted_suffixes_list):
        raise ValueError("The true suffixes and predicted suffixes must have the same length.")

    total_accuracy = 0

    for true_suffix, predicted_suffix in zip(true_suffixes_list, predicted_suffixes_list):
        # Create a mapping of events to unique characters
        event_mapping = {event: str(i) for i, event in enumerate(set(detokenize_suffix(true_suffix, tokenizer) + detokenize_suffix(predicted_suffix, tokenizer)))}
        
        # Apply the mapping to convert event sequences into "strings"
        true_str = ''.join([event_mapping[event] for event in true_suffix])
        predicted_str = ''.join([event_mapping[event] for event in predicted_suffix])
        
        # Calculate the Damerau-Levenshtein distance
        distance = damerau_levenshtein_distance(true_str, predicted_str)
        
        # Normalize the distance to get a score between 0 and 1
        max_len = max(len(true_str), len(predicted_str))
        normalized_distance = distance / max_len if max_len > 0 else 0
        
        # Calculate accuracy for the pair and accumulate
        accuracy = 1 - normalized_distance
        total_accuracy += accuracy

    # Compute the average accuracy across all suffixes
    average_accuracy = total_accuracy / len(true_suffixes_list) if true_suffixes_list else 0
    return average_accuracy

def evaluate_model(model, X, Y, tokenizer):
        # Predict the sequences
        predictions = model.predict(X)
        print(predictions.shape)

        # Convert predictions to sequence of integers
        predictions_seq = np.argmax(predictions, axis=-1)

        # Convert true values to sequence of integers
        Y_seq = np.argmax(Y, axis=-1)

        # Calculate MAE and MSE
        mae = mean_absolute_error(Y_seq.flatten(), predictions_seq.flatten())
        mse = mean_squared_error(Y_seq.flatten(), predictions_seq.flatten())
        
        # Calculate accuracy using damerau-leenshtein distance
        acc = evaluate_event_suffix_predictions(Y, predictions)

        return mae, mse, acc

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
    mae, mse, acc = evaluate_model(model, X_test_pad, Y_test_cat, tokenizer)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Accuracy in terms of Damerau-Levenshtein distance: {acc}")