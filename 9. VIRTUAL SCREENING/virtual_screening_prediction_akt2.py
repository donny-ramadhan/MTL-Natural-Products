import pandas as pd
import numpy as np
import joblib

# Load dataset
file_path = '/Users/Ramadhan/NP-MTL/Virtual_Screeing/natural_product_structures_final_data.txt'
data = pd.read_csv(file_path, sep="\t", low_memory=False)

# Identify 'Sim_*' columns dynamically
sim_columns = [col for col in data.columns if col.startswith('Sim_')]

# Process 'Avalon_FP' values
data['Avalon_FP'] = data['Avalon_FP'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

# Include 'Target ChEMBL ID' as a feature, placing it before 'sim_columns'
data['Features'] = data.apply(lambda row: np.concatenate(
    [[row['Target ChEMBL ID']], row[sim_columns].dropna().values.astype(float), row['Avalon_FP']]), axis=1)

# Prepare feature matrix X
X = np.stack(data['Features'].values)

# Load the trained Random Forest Regressor model
model_path = '/Users/Ramadhan/NP-MTL/Virtual_Screening/virtual_screening_akt2_model.joblib'
model = joblib.load(model_path)

# Predict pIC50 values
Prediction = model.predict(X)

# Add predictions to the DataFrame
data['Prediction'] = Prediction

# Save the predictions to a new file
output_file = '/Users/Ramadhan/NP-MTL/Virtual_Screeing/natural_product_activity_akt2_prediction.txt'
data.to_csv(output_file, sep='\t', index=False)