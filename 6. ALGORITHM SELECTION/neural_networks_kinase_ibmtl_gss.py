import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Load dataset
file_path = '/Users/Ramadhan/NP-MTL/Filtering_Preprocessing/kinase_final_dataset_gss_for_stl_mtl.txt'
data = pd.read_csv(file_path, sep="\t", low_memory=False)

# Identify 'Sim_*' columns dynamically
sim_columns = [col for col in data.columns if col.startswith('Sim_')]

# Process 'Avalon_FP' values
data['Avalon_FP'] = data['Avalon_FP'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

# Include 'Target ChEMBL ID' as a feature, placing it before the sim_columns
data['Features'] = data.apply(lambda row: np.concatenate(
    [[row['Target ChEMBL ID']], row[sim_columns].dropna().values.astype(float), row['Avalon_FP']]), axis=1)

# Prepare columns for predictions and fold information
data['Prediction'] = np.nan
data['Fold'] = np.nan

# Dictionary to store best hyperparameters per outer fold
best_params_dict = {}

# Prepare feature matrix X and target vector y
X = np.stack(data['Features'].values)
y = data['pIC50'].values

# Process nested CV using an outer StratifiedKFold split
outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
fold_num = 0

for outer_train_index, outer_test_index in outer_skf.split(X, data['Target ChEMBL ID']):
    X_outer_train, X_outer_test = X[outer_train_index], X[outer_test_index]
    y_outer_train, y_outer_test = y[outer_train_index], y[outer_test_index]
    
    # Set up inner CV on the outer training set (using the same stratification)
    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    stratify_labels = data.loc[outer_train_index, 'Target ChEMBL ID']
    
    # Define the objective function for inner hyperparameter tuning (using RMSE)
    def objective(trial):
        n_units = trial.suggest_int("n_units", 16, 128)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 50, 200)
        batch_size = trial.suggest_int("batch_size", 16, 128)

        rmse_scores = []
        for inner_train_index, inner_val_index in inner_skf.split(X_outer_train, stratify_labels):
            X_train_fold = X_outer_train[inner_train_index]
            y_train_fold = y_outer_train[inner_train_index]
            X_val_fold = X_outer_train[inner_val_index]
            y_val_fold = y_outer_train[inner_val_index]

            # Build the Keras model
            model = Sequential()
            model.add(Input(shape=(X_train_fold.shape[1],)))
            for i in range(n_layers):
                model.add(Dense(n_units, activation="relu"))
            model.add(Dense(1))
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss="mse")
            
            # Train the model (suppressing verbose output)
            model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Predict on the validation fold and calculate RMSE
            y_pred = model.predict(X_val_fold).flatten()
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)
    
    # Optimize hyperparameters using Optuna on the inner CV
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    best_params_dict[f"fold_{fold_num}"] = best_params
    
    # Train the final model on the entire outer training set using the best inner parameters
    final_model = Sequential()
    final_model.add(Input(shape=(X_outer_train.shape[1],)))
    for i in range(best_params["n_layers"]):
        final_model.add(Dense(best_params["n_units"], activation="relu"))
    final_model.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"])
    final_model.compile(optimizer=optimizer, loss="mse")
    
    final_model.fit(X_outer_train, y_outer_train, epochs=best_params["epochs"],
                    batch_size=best_params["batch_size"], verbose=0)
    
    # Predict on the outer test set
    y_outer_pred = final_model.predict(X_outer_test).flatten()
    
    # Save predictions and fold number in the original DataFrame
    data.loc[outer_test_index, 'Prediction'] = y_outer_pred
    data.loc[outer_test_index, 'Fold'] = fold_num
    
    fold_num += 1

# Calculate outer test performance (RMSE) for each unique group
results = []
group_columns = ['Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID', 'Fold']
for id_group, group_data in data.groupby(group_columns):
    true_values = group_data['pIC50']
    predicted_values = group_data['Prediction']
    
    # Skip groups with NaN or infinite values
    if np.any(np.isnan(true_values)) or np.any(np.isnan(predicted_values)):
        print(f"NaN values found in {id_group}, skipping this group.")
        continue
    if np.any(np.isinf(true_values)) or np.any(np.isinf(predicted_values)):
        print(f"Infinite values found in {id_group}, skipping this group.")
        continue

    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    
    results.append({
        'Target Parent ID': id_group[0],
        'Protein Class ID': id_group[1],
        'Target ChEMBL ID': id_group[2],
        'Fold': id_group[3],
        'RMSE': rmse
    })

# Sort the predictions data as specified
data.sort_values(by=['Fold', 'Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID'], inplace=True)

# Save the predictions file
data_save_path = '/Users/Ramadhan/NP-MTL/Algorithm_Selection/neural_networks_kinase_ibmtl_gss_prediction.txt'
data.to_csv(data_save_path, sep='\t', index=False)

# Convert the results list to a DataFrame and compute grouped statistics (outer performance) per target
results_df = pd.DataFrame(results)
final_results = results_df.groupby(['Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID']).agg({
    'RMSE': ['mean', 'std']
}).reset_index()
final_results.columns = [
    'Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID', 
    'IBMTL AA-GSS Average RMSE', 'IBMTL AA-GSS RMSE SD'
]

# Merge with 'Average Similarity' from the input file
average_similarity = data[['Target Parent ID', 'Protein Class ID', 'Average Similarity']].drop_duplicates()
final_results = final_results.merge(average_similarity, on=['Target Parent ID', 'Protein Class ID'], how='left')

# Save the statistic file
final_results_save_path = '/Users/Ramadhan/NP-MTL/Algorithm_Selection/neural_networks_kinase_ibmtl_gss_statistics.txt'
final_results.to_csv(final_results_save_path, sep='\t', index=False)
