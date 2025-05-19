import pandas as pd

# Read the tab-delimited file into a DataFrame
df = pd.read_csv("/Users/Ramadhan/NP-MTL/Filtering_Preprocessing/kinase_three_groups_target_parent_final_dataset_gss_for_stl_mtl", sep="\t", low_memory=False)

# Filter the DataFrame for rows where 'Target Parent ID' equals 1289.
# Adjust the type (int or string) as needed; here we assume it's numeric.
filtered_df = df[df["Target Parent ID"] == 1289]

# Define the list of columns to remove: "Sim_4" to "Sim_34"
cols_to_remove = [f"Sim_{i}" for i in range(4, 35)]

# Drop the specified columns. Using errors='ignore' in case some columns are missing.
filtered_df = filtered_df.drop(columns=cols_to_remove, errors='ignore')

# Write the filtered DataFrame to a new tab-delimited file
filtered_df.to_csv("/Users/Ramadhan/NP-MTL/Virtual_Screening/kinase_three_groups_target_parent_final_dataset_gss_for_virtual_screening.txt", sep="\t", index=False)
