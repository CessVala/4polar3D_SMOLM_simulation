"""
runBatch.py

Batch processing script for executing a Jupyter notebook on multiple datasets.

This script reads a list of filenames from an Excel file and uses Papermill to
execute a specified Jupyter notebook for each file. The notebook is expected to
process the data corresponding to each filename.

Author: Cesar Valades-Cruz, IHB
Date: April 29, 2025
"""


import papermill as pm
import pandas as pd


# Define the directory containing the Excel file and the notebook
dir_='Synthetic_images/'

# Read the Excel file containing the list of filenames
df=pd.read_excel(dir_+'listnames.xlsx')
filenames=df['Filenames'].tolist()

# Iterate over each selected filename
for filename in filenames:
    # Execute the specified Jupyter notebook with the current filename as a parameter    
    pm.execute_notebook(
        'ComparingFitting.ipynb',
        output_path=None,
        parameters={'filename': filename,'dir_':dir_}
    )
