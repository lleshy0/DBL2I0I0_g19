import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
import pm4py
from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import tool2_final as tool

def plot_conf_matrix(y_true, y_pred, title):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, include_values=False, xticks_rotation='vertical')
    plt.title(title)
    plt.show()
    

if __name__ == "__main__":
    filepath = 'naive_prediction.csv'
    prediction_df = pd.read_csv(filepath)
    y_true = prediction_df['next_event']
    y_pred = prediction_df['naive_prediction_action']
    plot_conf_matrix(y_true, y_pred)
    
