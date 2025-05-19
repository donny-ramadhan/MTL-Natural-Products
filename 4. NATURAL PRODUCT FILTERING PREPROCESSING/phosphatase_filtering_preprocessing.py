import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import SaltRemover, MolFromSmiles, MolToSmiles
from rdkit.Avalon import pyAvalonTools

# Load the data from the input file
file_path = '/Users/Ramadhan/NP-MTL/Filtering_Preprocessing/phosphatase_datasets.tsv'
data = pd.read_csv(file_path, sep='\t', encoding='windows-1252', low_memory=False)

# Split the rows based on "Standard Units"
nM_df = data[data['Standard Units'] == 'nM']
not_nM_df = data[data['Standard Units'] != 'nM']

# For the non-nM units, further split based on 'ug.mL-1'
ug_ml_df = not_nM_df[not_nM_df['Standard Units'] == 'ug.mL-1'].copy()
not_ug_ml_df = not_nM_df[not_nM_df['Standard Units'] != 'ug.mL-1'].copy()

# Convert 'Standard Value' for 'ug.mL-1' to 'nM'
ug_ml_df.loc[:, 'Standard Value'] = (ug_ml_df['Standard Value'] / ug_ml_df['Molecular Weight']) * 1e6
ug_ml_df.loc[:, 'Standard Units'] = 'nM'

# Concatenate the modified DataFrames
data = pd.concat([nM_df, ug_ml_df])

# Remove rows with 'Standard Value' = 0
data = data[data['Standard Value'] != 0]

# Get unique combinations of the identifiers
unique_combinations = data[['Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID']].drop_duplicates()

# Initialize the salt remover
remover = SaltRemover.SaltRemover()

# Load the best natural product filter model
model_path = '/Users/Ramadhan/NP-MTL/Binary_Classification/best_natural_product_filter.joblib'
rf_model = joblib.load(model_path)

# Function to process SMILES and calculate Avalon fingerprint
def process_smiles(smiles):
    try:
        mol = MolFromSmiles(smiles)
        if mol:
            mol = remover.StripMol(mol)
            cleaned_smiles = MolToSmiles(mol)
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=1024)
            return [cleaned_smiles, list(fp)]
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
    return [None, None]

# List to store each DataFrame
all_data_frames = []

# Iterate over each unique combination of IDs
for index, combo in unique_combinations.iterrows():
    filtered_data = data[
        (data['Target Parent ID'] == combo['Target Parent ID']) &
        (data['Protein Class ID'] == combo['Protein Class ID']) &
        (data['Target ChEMBL ID'] == combo['Target ChEMBL ID']) &
        (data['Standard Type'] == 'IC50')
    ]

    # Drop rows with missing values in 'SMILES', 'Standard Value', and 'Standard Relation'
    filtered_data = filtered_data.dropna(subset=['SMILES', 'Standard Value', 'Standard Relation'])

    # Handle missing values in 'Document Year' by filling them with a very low value (e.g., 0)
    filtered_data['Document Year'] = filtered_data['Document Year'].fillna(0)

    # Sort the DataFrame first by 'Molecule ChEMBL ID', then by 'Document Year' in descending order
    # and 'Standard Value' in ascending order to prepare for deduplication
    filtered_data.sort_values(by=['Molecule ChEMBL ID', 'Document Year', 'Standard Value'], ascending=[True, False, True], inplace=True)

    # Remove duplicates: keeping the first entry for each 'Molecule ChEMBL ID' which will be
    # the one with the maximum 'Document Year' and if tied, the minimum 'Standard Value'
    filtered_data = filtered_data.drop_duplicates(subset='Molecule ChEMBL ID', keep='first')
        
    # Apply the process_smiles function and join results
    results = filtered_data['SMILES'].apply(lambda x: process_smiles(x))
    results_df = pd.DataFrame(results.tolist(), index=results.index, columns=['Cleaned_SMILES', 'Avalon_FP'])
    filtered_data = filtered_data.join(results_df)

    # Prepare the Avalon fingerprints for prediction
    valid_fps = filtered_data['Avalon_FP'].dropna().tolist()
    valid_fps = np.array([np.array(fp) for fp in valid_fps if fp is not None])

    # Predict using the best natural product filter model
    if len(valid_fps) > 0:
        predictions = rf_model.predict(valid_fps)
        filtered_data.loc[filtered_data['Avalon_FP'].notnull(), 'Predictions'] = predictions

    # Append to the list
    all_data_frames.append(filtered_data)

# Concatenate all DataFrames
final_df = pd.concat(all_data_frames)

# Filter for rows with prediction 'NC'
final_df = final_df[final_df['Predictions'] == 'NC']

# Filter for unique combinations with at least 50 entries
final_filtered_df = final_df.groupby(['Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID']).filter(lambda x: len(x) >= 50)

# Extract the numeric part of "Molecule ChEMBL ID" and convert it to an integer for final sorting
final_filtered_df['Molecule ChEMBL ID Numeric'] = final_filtered_df['Molecule ChEMBL ID'].str.extract('(\d+)').astype(int)

# Sort by 'Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID', and the numeric part of 'Molecule ChEMBL ID'
final_filtered_df.sort_values(by=['Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID', 'Molecule ChEMBL ID Numeric'], inplace=True)

# Drop the temporary numeric ID column
final_filtered_df.drop(columns=['Molecule ChEMBL ID Numeric'], inplace=True)

# Filter rows with 'Standard Relation' of '='
input_table = final_filtered_df[final_filtered_df['Standard Relation'] == '=']

# Filter for unique combinations with at least 50 entries
filtered_table = input_table.groupby(['Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID']).filter(lambda x: len(x) >= 50)

# Convert 'Standard Value' to 'pIC50' and add it as a new column
filtered_table['pIC50'] = -np.log10(filtered_table['Standard Value'] * 10**-9)

# Remove the prefix 'CHEMBL' in 'Target ChEMBL ID'
filtered_table['Target ChEMBL ID'] = filtered_table['Target ChEMBL ID'].str.replace('CHEMBL', '', regex=False)

# Save the final DataFrame to a new text file with tab separation
output_file_path = '/Users/Ramadhan/NP-MTL/Filtering_Preprocessing/phosphatase_final_dataset_for_stl_mtl.txt'
filtered_table.to_csv(output_file_path, index=False, sep='\t')
