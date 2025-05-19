import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import optuna

# Load dataset
file_path = '/Users/Ramadhan/NP-MTL/Filtering_Preprocessing/ligase_final_dataset_lss_for_stl_mtl.txt'
data = pd.read_csv(file_path, sep="\t", low_memory=False)

# Identify 'Sim_*' columns dynamically
sim_columns = [col for col in data.columns if col.startswith('Sim_')]

# Process 'Avalon_FP' values
data['Avalon_FP'] = data['Avalon_FP'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

# Include 'Target ChEMBL ID' as a feature, placing it before 'sim_columns'
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

# Process nested CV
outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
fold_num = 0

for outer_train_index, outer_test_index in outer_skf.split(X, data['Target ChEMBL ID']):
    X_outer_train, X_outer_test = X[outer_train_index], X[outer_test_index]
    y_outer_train, y_outer_test = y[outer_train_index], y[outer_test_index]
    
    # Set up inner CV on the outer training set (using the same stratification)
    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    stratify_labels = data.loc[outer_train_index, 'Target ChEMBL ID']
    
    # Define the objective function for inner hyperparameter tuning (using only RMSE)
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 25)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=100
        )
        
        inner_rmse_scores = []
        for inner_train_index, inner_val_index in inner_skf.split(X_outer_train, stratify_labels):
            X_inner_train = X_outer_train[inner_train_index]
            y_inner_train = y_outer_train[inner_train_index]
            X_inner_val = X_outer_train[inner_val_index]
            y_inner_val = y_outer_train[inner_val_index]
            
            model.fit(X_inner_train, y_inner_train)
            y_inner_pred = model.predict(X_inner_val)
            inner_rmse = np.sqrt(mean_squared_error(y_inner_val, y_inner_pred))
            inner_rmse_scores.append(inner_rmse)
        
        return np.mean(inner_rmse_scores)
    
    # Optimize hyperparameters using Optuna on the inner CV
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    best_params_dict[f"fold_{fold_num}"] = best_params
    
    # Train the final model on the entire outer training set with the best inner parameters
    final_model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        random_state=100
    )
    final_model.fit(X_outer_train, y_outer_train)
    
    # Predict on the outer test set
    y_outer_pred = final_model.predict(X_outer_test)
    
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
data_save_path = '/Users/Ramadhan/NP-MTL/STL_MTL_Studies/ligase_rf_ibmtl_lss_prediction.txt'
data.to_csv(data_save_path, sep='\t', index=False)

# Convert the results list to a DataFrame and compute grouped statistics (outer performance) per target
results_df = pd.DataFrame(results)
final_results = results_df.groupby(['Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID']).agg({
    'RMSE': ['mean', 'std']
}).reset_index()
final_results.columns = [
    'Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID', 
    'IBMTL AA-LSS Average RMSE', 'IBMTL AA-LSS RMSE SD'
]

# Merge with 'Average Similarity' from the input file
average_similarity = data[['Target Parent ID', 'Protein Class ID', 'Average Similarity']].drop_duplicates()
final_results = final_results.merge(average_similarity, on=['Target Parent ID', 'Protein Class ID'], how='left')

# Save the statistic file
final_results_save_path = '/Users/Ramadhan/NP-MTL/STL_MTL_Studies/ligase_rf_ibmtl_lss_statistics.txt'
final_results.to_csv(final_results_save_path, sep='\t', index=False)
