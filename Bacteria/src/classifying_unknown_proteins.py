#!/usr/bin/env python3
# coding: utf-8

########################################################################################################################
# Author: Tanja Krüger
# (Modified by Gemini to enforce FASTA file order and handle ID mismatches)
# Aim: This file reuses a finished trained predictor for new predictions, ensuring output order matches the input FASTA.
#      It specifically handles cases where '.' in FASTA IDs is converted to '_' in H5 file keys.
# Input1: a trained predictor.
#  Where:You can find those under the Predictor folder of this project.
#  Recommended model: sklearn_svcPC20_SST30_CV10_embeddingsProtT5.
#  Alternatives: The same folder has other alternatives. Everything that starts with sklearn in the beginning of the
#      name is compatible with this script. Careful what model you choose, some were trained on scrambled sequences.
#      Predictions will be significantly less good - these models have "allscr" in their name.
# Input2: X: embeddings from the sequences you want to predict (in .h5 format).
# Input3: The original FASTA file used to generate the embeddings (to define the output order).
# Output: The predictions in a .jsonl file, with order corresponding to the FASTA file.

########################################################################################################################

import numpy as np
import pickle
import h5py
import pandas as pd
import re, argparse, json, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from metrics import write_metric
from collections import Counter
from datetime import datetime

# #################################################################################################
# Option depending where the user wants the run the code form, default running the code with make from the project folder.
cl=""
# If one wants to execute this file from the Code/python folder uncomment the next line.
#cl="../../"

########################################################################################################################
# Get the arguments from the command line.
parser = argparse.ArgumentParser(
    prog="classifying_unknown_proteins",
    description="Predicts toxicity for proteins from their embeddings, ensuring the output order matches the original FASTA file."
)
parser.add_argument("pred",
                    type=str,
                    help="Path to the trained predictor file.")
parser.add_argument("X",
                    type=str,
                    help="Path to the protein embeddings file in .h5 format.")
parser.add_argument("fasta",
                    type=str,
                    help="Path to the original FASTA file to establish the correct output order.")
parser.add_argument("--metric_file", type=str, default=None, help="指标输出 JSONL 文件")

args = parser.parse_args()

# Extract information from the provided parameters (optional, for metadata)
try:
    cv = re.search("CV(\d+)_", args.pred).group(1)
    where = re.search("Predictor/(.*)_(.*)_SST(\d+)_", args.pred).group(1)
    sst_level = re.search("SST(\d+)", args.pred).group(1)
    embedding_type_train = re.search("Predictor/(.*)_(.*)_SST(\d+)_CV(\d+)_(.*)", args.pred).group(5)
    architecture = re.search("Predictor/(.*)_(.*)_SST(\d+)_", args.pred).group(2)
except AttributeError:
    print("Warning: Could not parse metadata from predictor filename. Continuing with prediction.")
    pass


########################################################################################################################
# Step 1: Open data and prepare it in the correct order

# Step 1.1: Read protein IDs from the FASTA file to get the desired order
print(f"Reading protein ID order from: {args.fasta}")
protein_ids_in_fasta_order = []
try:
    with open(args.fasta, 'r') as fasta_file:
        for line in fasta_file:
            if line.startswith('>'):
                # Extract the ID, which is typically the first word after '>'
                protein_id = line[1:].strip().split()[0]
                protein_ids_in_fasta_order.append(protein_id)
except FileNotFoundError:
    print(f"Error: The specified FASTA file was not found at '{args.fasta}'")
    exit(1)

if not protein_ids_in_fasta_order:
    print(f"Error: No protein IDs found in the FASTA file '{args.fasta}'. Please check the file format.")
    exit(1)

print(f"Found {len(protein_ids_in_fasta_order)} protein IDs in FASTA file.")

# Step 1.2: Open the predictor
with open(args.pred, 'rb') as f:
    predictor = pickle.load(f)

# Step 1.3: Open the embeddings and load data according to the FASTA order
print(f"Loading embeddings from: {args.X}")
with h5py.File(args.X, "r") as f:
    # Create a mapping from simple ID (already normalized in H5) to the full key for efficient lookup
    h5_key_map = {key.split(' ', 1)[0]: key for key in f.keys()}

    embeddings_ordered = []
    final_ordered_ids = []
    
    # Iterate through the IDs from the FASTA file to build our dataset in the correct order
    for original_fasta_id in protein_ids_in_fasta_order:
        # NORMALIZE the FASTA ID for matching: replace '.' with '_'
        # This handles cases like 'WP_013097732.1' in FASTA becoming 'WP_013097732_1' in H5.
        normalized_id_for_lookup = original_fasta_id.replace('.', '_')

        if normalized_id_for_lookup in h5_key_map:
            full_h5_key = h5_key_map[normalized_id_for_lookup]
            embeddings_ordered.append(list(f[full_h5_key]))
            # IMPORTANT: We use the ORIGINAL fasta ID for the output, not the normalized one.
            final_ordered_ids.append(original_fasta_id)
        else:
            # If an ID from the FASTA file is still not found after normalization, print a warning.
            print(f"Warning: ID '{original_fasta_id}' not found in H5 embeddings file (even after normalization). It will be skipped.")
    
    # Create the pandas DataFrame with the data now in the correct order and with original IDs.
    X = pd.DataFrame(embeddings_ordered, index=final_ordered_ids)

if X.empty:
    print("Error: No matching protein IDs found between the FASTA file and the H5 file. Cannot proceed.")
    # 写指标并退出
    write_metric(args.metric_file, "bacteria_classify", {
        "pred_model": args.pred,
        "h5": args.X,
        "fasta": args.fasta,
        "num_predictions": 0,
        "error": "no_matching_ids",
    })
    exit(1)

########################################################################################################################
# Step 2: Run the prediction on the correctly ordered data
print("\nRunning predictions...")
y_pred = predictor.predict(X)
y_probas = predictor.predict_proba(X)[:, 1]
count = Counter(y_pred)

print("Prediction counts:", count)

########################################################################################################################
# Step 3: Save the results to a JSONL file, preserving the order
output_filename = os.path.splitext(args.X)[0] + '.jsonl'

print(f"\nSaving results to: {output_filename}")
try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Zip together the correctly ordered (original) IDs, predictions, and probabilities
        for protein_id, prediction, proba in zip(X.index, y_pred, y_probas):
            result_dict = {
                'id': protein_id,
                # Convert numpy types to standard Python types for JSON serialization
                'prediction': int(prediction),
                'probability': float(proba)
            }
            # Write the dictionary as a JSON string on a new line
            f.write(json.dumps(result_dict) + '\n')
    
    print(f"Successfully saved {len(X)} predictions.")
    # 写指标
    try:
        pred_counts = {str(int(k)): int(v) for k, v in count.items()}
    except Exception:
        pred_counts = {}
    write_metric(args.metric_file, "bacteria_classify", {
        "pred_model": args.pred,
        "h5": args.X,
        "fasta": args.fasta,
        "num_predictions": int(len(X)),
        "class_counts": pred_counts,
        "output_jsonl": output_filename,
    })

except Exception as e:
    print(f"\nAn error occurred while writing to the file: {e}")