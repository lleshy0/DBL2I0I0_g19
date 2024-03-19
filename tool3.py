import rf_event_prediction as ep
import pm4py as pm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from pm4py.objects.conversion.log import converter as xes_converter
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
from scipy import stats
import warnings
import pickle
import event_conf_matrix as mat
import plot_features as pt

def xes_to_df(file_path):
    event_log = pm.read_xes(file_path)
    event_df = pm.convert_to_dataframe(event_log)

    return event_df

def event_prediction(test_df):   
    # import event prediction model
    event_model_pkl_file = 'event_model.pkl'
    with open(event_model_pkl_file, 'rb') as file:  
        model = pickle.load(file)
    
    features = ep.prep_features(test_df)
    true_labels = ep.append_next_event(test_df)
    features.to_csv("test_features.csv")
    
    # run prediction
    pred_labels = pd.DataFrame(model.predict(features))
    pred_labels.columns = ep.CATEGORIES
    pred_labels = pred_labels.idxmax(axis=1)
    
    pt.plot_feature_importance(model, features.columns)
    
    return true_labels, pred_labels
       
def time_prediction():
    # import model
    return 0

if __name__ == "__main__":
    # import test data
    file_path = "test.xes"
    test_df = pm.read_xes(file_path)
    
    # evaluate event prediction on test set
    true_labels, pred_labels = event_prediction(test_df)
    
    print("Event prediction accuracy: ")
    print(ep.evaluate_event_prediction(true_labels, pred_labels))
    