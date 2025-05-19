import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Avalon import pyAvalonTools
from sklearn.ensemble import RandomForestClassifier
import joblib

# Read the first file
file1_path = '/Users/Ramadhan/NP-MTL/Binary_Classification/ibs2023mar_nc.smi'
file1_df = pd.read_csv(file1_path, header=None, delimiter='\t')

# Read the second file
file2_path = '/Users/Ramadhan/NP-MTL/Binary_Classification/ibs2023mar_sc1.smi'
file2_df = pd.read_csv(file2_path, header=None, delimiter='\t')

# Randomly select 10000 rows from each file
random_state = 100
file1_sampled = file1_df.sample(n=10000, random_state=random_state)
file2_sampled = file2_df.sample(n=10000, random_state=random_state)

# Concatenate the sampled dataframes
concatenated_df = pd.concat([file1_sampled, file2_sampled])

# Function to convert SMILES to Canonical SMILES and remove salts, and then convert to Avalon fingerprints
def convert_to_canonical_smiles_and_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Convert to canonical SMILES
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        # Remove salts
        remover = SaltRemover.SaltRemover()
        mol_without_salt = remover.StripMol(mol)
        if mol_without_salt is not None:
            canonical_smiles = Chem.MolToSmiles(mol_without_salt, isomericSmiles=False)
            # Generate Avalon fingerprints with 1024 bits
            fingerprint = pyAvalonTools.GetAvalonFP(mol_without_salt, nBits=1024)
            return canonical_smiles, fingerprint
        else:
            return canonical_smiles, None  # Return original SMILES and None if salt stripping fails
    else:
        return None, None

# Apply the conversion function to the first column of the concatenated dataframe
concatenated_df[[0, 2]] = concatenated_df[0].apply(lambda x: pd.Series(convert_to_canonical_smiles_and_fingerprint(x)))

# Add third column based on conditions
pattern_nc = re.compile(r'STOCK.N.')
pattern_sc = re.compile(r'STOCK.S.')

concatenated_df[3] = concatenated_df[1].apply(lambda x: 'NC' if pattern_nc.match(x) else 'SC')

# Initialize parameters for the Random Forest Classifier
n_estimators = 100

# Initialize a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

# Train the Random Forest Classifier on all data
X_train = list(concatenated_df[2])
y_train = concatenated_df[3]
rf_classifier.fit(X_train, y_train)

# Save the trained model
model_path = '/Users/Ramadhan/NP-MTL/Binary_Classification/best_natural_product_filter.joblib'
joblib.dump(rf_classifier, model_path)