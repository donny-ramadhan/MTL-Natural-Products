import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import optuna

# Load dataset
file_path = '/Users/Ramadhan/NP-MTL/Filtering_Preprocessing/kinase_three_groups_final_dataset_for_stl_mtl.txt'
data = pd.read_csv(file_path, sep="\t", low_memory=False)

# Process 'Avalon_FP' values
data['Avalon_FP'] = data['Avalon_FP'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

# Include 'Target ChEMBL ID' as a feature, placing it before 'Avalon_FP'
data['Features'] = data.apply(lambda row: np.concatenate([[row['Target ChEMBL ID']], row['Avalon_FP']]), axis=1)

# Prepare columns for predictions and fold information
data['Prediction'] = np.nan
data['Fold'] = np.nan

# Assign global fold numbers using StratifiedKFold on the entire dataset (stratified by 'Target ChEMBL ID')
global_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
data['GlobalFold'] = np.nan
for fold, (train_index, test_index) in enumerate(global_skf.split(data, data['Target ChEMBL ID'])):
    data.loc[test_index, 'GlobalFold'] = fold

# Dictionary to store best hyperparameters per group and fold
best_params_dict = {}

# Group by Target Parent ID (excluding Protein Class ID)
groups = data.groupby(['Target Parent ID'])

# Process nested CV for each group separately using the pre-assigned global folds
for target_parent_id, group in groups:
    # Prepare feature matrix X and target vector y for the group
    X_group = np.stack(group['Features'].values)
    y_group = group['pIC50'].values
    
    # Get the unique global fold values present in the group
    group_global_folds = group['GlobalFold'].unique()
    
    for fold in group_global_folds:
        # Identify outer test and train indices for the group based on the pre-assigned GlobalFold
        outer_test_index = group.index[group['GlobalFold'] == fold]
        outer_train_index = group.index[group['GlobalFold'] != fold]
        
        # Get positions within the group (since X_group is a numpy array following the group order)
        outer_test_positions = np.where(group['GlobalFold'] == fold)[0]
        outer_train_positions = np.where(group['GlobalFold'] != fold)[0]
        
        X_outer_train = X_group[outer_train_positions]
        y_outer_train = y_group[outer_train_positions]
        X_outer_test = X_group[outer_test_positions]
        y_outer_test = y_group[outer_test_positions]
        
        # For inner CV, use StratifiedKFold on the outer training set
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
        # Use 'Target ChEMBL ID' as stratification labels from the outer training set
        stratify_labels_inner = group.iloc[outer_train_positions]['Target ChEMBL ID']
        
        # Define the objective function for inner hyperparameter tuning (using RMSE)
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 3, 25)
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=100
            )
            
            inner_rmse_scores = []
            for inner_train_index, inner_val_index in inner_cv.split(X_outer_train, stratify_labels_inner):
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
        best_params_dict[(target_parent_id, fold)] = best_params
        
        # Train the final model on the entire outer training set using the best inner parameters
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
        data.loc[outer_test_index, 'Fold'] = fold

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

# Sort the prediction data as specified
data.sort_values(by=['Fold', 'Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID'], inplace=True)

# Save the predictions file
data_save_path = '/Users/Ramadhan/NP-MTL/STL_MTL_Three_Groups_Studies/kinase_three_groups_target_parent_rf_fbmtl_prediction.txt'
data.to_csv(data_save_path, sep='\t', index=False)

# Convert the results list to a DataFrame and compute grouped statistics (outer performance) per target
results_df = pd.DataFrame(results)
final_results = results_df.groupby(['Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID']).agg({
    'RMSE': ['mean', 'std']
}).reset_index()
final_results.columns = [
    'Target Parent ID', 'Protein Class ID', 'Target ChEMBL ID', 
    'FBMTL Average RMSE', 'FBMTL RMSE SD'
]

# Save the statistic file
final_results_save_path = '/Users/Ramadhan/NP-MTL/STL_MTL_Three_Groups_Studies/kinase_three_groups_target_parent_rf_fbmtl_statistics.txt'
final_results.to_csv(final_results_save_path, sep='\t', index=False)
