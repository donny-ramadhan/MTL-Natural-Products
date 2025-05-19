import pandas as pd
from Bio import Align, SeqIO
from Bio.Align import substitution_matrices, PairwiseAligner
from io import StringIO
import requests

# Function to fetch protein sequence from UniProt
def fetch_protein_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        record = SeqIO.read(StringIO(fasta_data), "fasta")
        return str(record.seq)
    else:
        print(f"Failed to fetch sequence for {uniprot_id}")
        return None

# Function to calculate similarity between two sequences using PairwiseAligner
def calculate_percentage_similarity(alignment, seq1, seq2):
    seqA, seqB = alignment.aligned
    alignment_length = 0
    similarity_count = 0

    for (startA, endA), (startB, endB) in zip(seqA, seqB):
        for i, j in zip(range(startA, endA), range(startB, endB)):
            if seq1[i] == seq2[j]:
                similarity_count += 1
            alignment_length += 1

    return (similarity_count / alignment_length) if alignment_length > 0 else 0

# Function to calculate average similarity above the diagonal
def calculate_average_similarity_above_diagonal(matrix):
    values = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            values.append(matrix.iloc[i, j])
    return sum(values) / len(values) if values else None

# Load the BLOSUM62 matrix
matrix = substitution_matrices.load("BLOSUM62")

# Define the aligner and set gap penalties
aligner = PairwiseAligner()
aligner.substitution_matrix = matrix
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5
aligner.mode = 'global'

# Load data
data_path = '/Users/Ramadhan/NP-MTL/Filtering_Preprocessing/protease_three_groups_final_dataset_for_stl_mtl.txt'
df = pd.read_csv(data_path, sep='\t', low_memory=False)

# Extract unique combinations of 'Target Parent ID' and corresponding 'UniProt Accession'
grouped = df.groupby(['Target Parent ID'])['UniProt Accession'].unique()

# Initialize empty dictionary to store similarity matrices and maximum matrix size
similarity_matrices = {}
max_size = 0

# Process each group
for (target_parent_id), uniprot_ids in grouped.items():
    valid_ids = [uid for uid in uniprot_ids if fetch_protein_sequence(uid) is not None]
    similarity_matrix = {}
    for id1 in valid_ids:
        seq1 = fetch_protein_sequence(id1)
        similarities = {}
        for id2 in valid_ids:
            seq2 = fetch_protein_sequence(id2)
            if seq1 and seq2:
                alignments = aligner.align(seq1, seq2)
                similarity = calculate_percentage_similarity(alignments[0], seq1, seq2)
            else:
                similarity = None
            similarities[id2] = similarity
        similarity_matrix[id1] = similarities
    df_sim = pd.DataFrame(similarity_matrix)
    similarity_matrices[(target_parent_id)] = df_sim
    max_size = max(max_size, df_sim.shape[1])

# Find position to insert new columns before 'Avalon_FP'
avalon_fp_index = df.columns.get_loc('Avalon_FP')

# Insert new columns before 'Avalon_FP'
for i in range(max_size):
    df.insert(avalon_fp_index + i, f"Sim_{i+1}", None)

# Find position to insert 'Average Similarity' column before 'pIC50'
class_index = df.columns.get_loc('pIC50')

# Insert the 'Average Similarity' column before 'pIC50'
df.insert(class_index, 'Average Similarity', None)

# Populate the new columns with similarity values and calculate the average similarity
for (tpid), matrix in similarity_matrices.items():
    ids = matrix.columns
    for idx, row in df[(df['Target Parent ID'] == tpid)].iterrows():
        accession = row['UniProt Accession']
        if accession in ids:
            sim_values = matrix.loc[accession].tolist()
            average_similarity = calculate_average_similarity_above_diagonal(matrix)
            df.loc[idx, [f"Sim_{i+1}" for i in range(len(sim_values))]] = sim_values
            df.loc[idx, 'Average Similarity'] = average_similarity

# Save the updated DataFrame to the specified path
output_path = '/Users/Ramadhan/NP-MTL/Filtering_Preprocessing/protease_three_groups_target_parent_final_dataset_gss_for_stl_mtl.txt'
df.to_csv(output_path, sep='\t', index=False)
