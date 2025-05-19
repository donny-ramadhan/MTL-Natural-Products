import pandas as pd
import numpy as np

# Read the data from the file
file_path = '/Users/Ramadhan/NP-MTL/Filtering_Preprocessing/kinase_final_dataset_for_stl_mtl.txt'
input_table = pd.read_csv(file_path, sep='\t')

# Ensure each Protein Class ID has at least 2 different Target ChEMBL IDs
filtered_table = input_table.groupby('Protein Class ID').filter(lambda x: x['Target ChEMBL ID'].nunique() >= 2)

# Save the updated dataframe to a new file
output_file_path = '/Users/Ramadhan/NP-MTL/Filtering_Preprocessing/kinase_three_groups_final_dataset_for_stl_mtl.txt'
filtered_table.to_csv(output_file_path, sep='\t', index=False)