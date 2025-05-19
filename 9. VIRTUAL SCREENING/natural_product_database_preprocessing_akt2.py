import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover, MolFromSmiles, MolToSmiles
from rdkit.Avalon import pyAvalonTools

# Initialize the salt remover
remover = SaltRemover.SaltRemover()

# Function to process SMILES and calculate Avalon fingerprint
def process_smiles(smiles):
    try:
        mol = MolFromSmiles(smiles)
        if mol:
            mol = remover.StripMol(mol)  # Remove salts and clean the molecule
            cleaned_smiles = MolToSmiles(mol)
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=1024)  # Generate Avalon fingerprint
            return [cleaned_smiles, list(fp)]
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
    return [None, None]

# Read the tab-separated text file
file_path = '/Users/Ramadhan/NP-MTL/Virtual_Screeing/natural_product_structures.txt'
data = pd.read_csv(file_path, sep='\t', low_memory=False)

# Apply the Avalon fingerprint conversion to the "smiles" column
fingerprint_data = data['smiles'].apply(process_smiles)

# Split the result into two new columns: 'Cleaned_SMILES' and 'Avalon_FP'
data[['Cleaned_SMILES', 'Avalon_FP']] = pd.DataFrame(fingerprint_data.tolist(), index=data.index)

# Add the new "Target ChEMBL ID" column with a fixed value of "2431"
data.insert(data.columns.get_loc('Cleaned_SMILES') + 1, 'Target ChEMBL ID', '2431')

# List of updated values for the Sim columns
sim_values = [
    1, 0.8162839248434238, 0.7857142857142857
]

# Add the new "Sim_" columns with their respective values
for i, value in enumerate(sim_values, start=1):
    column_name = f'Sim_{i}'
    data[column_name] = value

# Move the 'Avalon_FP' column to the end
columns = [col for col in data.columns if col != 'Avalon_FP'] + ['Avalon_FP']
data = data[columns]

# Save the updated file
output_path = '/Users/Ramadhan/NP-MTL/Virtual_Screeing/natural_product_structures_final_data.txt'
data.to_csv(output_path, sep='\t', index=False)
