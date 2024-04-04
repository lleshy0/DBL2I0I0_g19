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
from collections import defaultdict

max_sequence_len = sp.max_sequence_len

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

# Prediction function
def predict_suffix(model, tokenizer, prefix, max_length):
    sequence = tokenizer.texts_to_sequences([prefix])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')
    
    predicted_suffix = []
    for _ in range(max_length):
        predictions = model.predict(sequence, verbose=0)
        next_event_token = np.argmax(predictions, axis=-1)[0]
        if next_event_token == 0:  # 0 is used as the padding token and should not be part of the prediction.
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
    
    # Prepare data
    test_sequences = sp.get_event_sequences(test_df)
    
    # Generating predictions for test data
    predicted_suffixes = predict_suffixes_for_test_data(model, tokenizer, test_sequences, max_sequence_len, 5)
    
    # Generate the positional accuracies
    positional_accuracies = calculate_positional_accuracies(model, tokenizer, test_sequences, max_sequence_len)
    
    # Extract positions and their corresponding accuracies
    positions = list(positional_accuracies.keys())
    accuracies = [positional_accuracies[position] for position in positions]
    
    # Displaying predictions for a few cases
    for case_id, (prefix, suffix) in list(predicted_suffixes.items())[:5]:  # Display first 5 cases
        print(f"Case ID: {case_id}")
        print("Prefix:", prefix)
        print("Predicted Suffix:", suffix)
        print("\n")
    
    #Print out the positional accuracies
    for position, accuracy in sorted(positional_accuracies.items()):
        print(f"Accuracy for position {position+1}: {accuracy:.4f}")
    