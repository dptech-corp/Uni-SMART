import os
import csv
import json
import random
from prompts.mol_gen_w_context_prompt import mol_gen_w_context_example, mol_gen_w_context_template

# Function to load SMILES from CSV
def load_smiles(filepath):
    smiles = []
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            smiles.append(row['SMILES'])
    return smiles

# Function to randomly sample 5 SMILES
def sample_smiles(smiles, num_samples=5):
    return random.sample(smiles, min(num_samples, len(smiles)))

# Generate synthetic data using the prompt template
def generate_synthetic_data(num_samples=5):
    smiles = load_smiles(f'./prompts/LIBRARY_SUGARS.csv')
    synthetic_data = []
    random.seed(42)

    for i in range(num_samples):
        sampled_smiles = sample_smiles(smiles)
        keywords_str = ', '.join(sampled_smiles)
        system_prompt = mol_gen_w_context_template[0]["system"].format(mol_gen_example=mol_gen_w_context_example)
        user_prompt = mol_gen_w_context_template[0]["user"].format(keywords=keywords_str)

        # print("-" * 20)
        # print("User Prompt:", user_prompt)
        # print("-" * 20)
        
        synthetic_data.append({
            "system": system_prompt,
            "user": user_prompt,
        })
    
    return synthetic_data
