# -*- coding: windows-1252 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
file_path = '/Users/Ramadhan/NP-MTL/Virtual_Screening/kinase_three_groups_target_parent_final_dataset_gss_for_virtual_screening.txt'
data = pd.read_csv(file_path, sep="\t", low_memory=False)

# Identify 'Sim_*' columns dynamically
sim_columns = [col for col in data.columns if col.startswith('Sim_')]

# Process 'Avalon_FP' values
data['Avalon_FP'] = data['Avalon_FP'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

# Include 'Target ChEMBL ID' as a feature, placing it before 'sim_columns'
data['Features'] = data.apply(lambda row: np.concatenate(
    [[row['Target ChEMBL ID']], row[sim_columns].dropna().values.astype(float), row['Avalon_FP']]), axis=1)

# Prepare feature matrix X and target vector y
X = np.stack(data['Features'].values)
y = data['pIC50'].values

# Define fixed hyperparameters
n_estimators = 199
max_depth = 17

# Train the Random Forest Regressor with fixed hyperparameters
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=100)
model.fit(X, y)

# Save the trained model to a joblib file
model_save_path = '/Users/Ramadhan/NP-MTL/Virtual_Screening/virtual_screening_akt2_model.joblib'
joblib.dump(model, model_save_path)