import pm4py
from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd

def convert_xes_to_csv(xes_file_path, csv_file_path):
    # Import XES file
    log = xes_importer.apply(xes_file_path)
    pd = xes_converter.apply(log, variant=xes_converter.Variants.TO_DATA_FRAME)

    # Save DataFrame to CSV file
    pd.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    xes_file_path = "C:\Henrique\TUE\YEAR2\Q3\dbl\BPI_Challenge_2012.xes.gz"
    csv_file_path = "C:\Henrique\TUE\YEAR2\Q3\dbl\csv_2012.csv"
    convert_xes_to_csv(xes_file_path, csv_file_path)