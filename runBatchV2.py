"""
runBatchV2.py

Batch processing script for executing a Jupyter notebook on multiple datasets.

This script reads a list of filenames from an Excel file and uses Papermill to
execute a specified Jupyter notebook for each file. The notebook is expected to
process the data corresponding to each filename.

Author: Cesar Valades-Cruz, IHB
Date: November 19, 2025
"""


import papermill as pm
import pandas as pd


# Define the directory containing the Excel file and the notebook
dir_='E:\Synthetic_images_aberrations/'

# Read the Excel file containing the list of filenames
df=pd.read_excel(dir_+'listnames_part_02.xlsx')
filenames=df['Filenames'].tolist()

failed_files = []   # to store any files that crash

# Iterate over each selected filename
for filename in filenames:
    print(f"\n>>> Running: {filename}")
    try:
        # Execute notebook with parameters
        pm.execute_notebook(
            'ComparingFitting_Version2.ipynb',
            output_path=None,
            parameters={'filename': filename, 'dir_': dir_}
        )
        print(f"✔️ SUCCESS: {filename}")

    except Exception as e:
        print(f"❌ ERROR with file: {filename}")
        print(f"    Reason: {e}")
        failed_files.append(filename)
        continue

# -----------------------------
# FINAL REPORT
# -----------------------------
print("\n========================================")
print("Batch processing finished.")
print("========================================")

if len(failed_files) == 0:
    print("🎉 All files processed successfully!")
else:
    print(f"⚠️ {len(failed_files)} file(s) failed:")
    for f in failed_files:
        print("   -", f)

    # Optional: save the list to Excel
    pd.DataFrame({'FailedFiles': failed_files}).to_excel(
        dir_ + 'failed_files.xlsx', index=False
    )
    print(f"\nFailed file list saved to: {dir_}failed_files.xlsx")
