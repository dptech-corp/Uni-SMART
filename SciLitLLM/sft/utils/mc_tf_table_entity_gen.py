import os
import json
import random
import numpy as np
import sys

# Ensure the current directory is part of the path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import task-specific templates and examples for synthetic data generation
from prompts.table_extract_prompt import table_extract_example, table_extract_template
from prompts.multiple_choice_prompt import multiple_choice_example, multiple_choice_template
from prompts.TF_prompt import TF_example, TF_template
from prompts.entity_extraction_prompt import entity_extraction_example, entity_extraction_template

# Function to load keywords and their frequencies from a word frequency table file
def load_keywords(filepath):
    """
    Load keywords and their frequencies from a file and normalize them to form probabilities.
    
    Parameters:
    filepath (str): Path to the word frequency table file.

    Returns:
    list: A list of keywords.
    list: A list of normalized probabilities corresponding to the keywords.
    """
    keywords = []
    probabilities = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("Word Frequency Table:"):
                continue
            word, frequency = line.split(": ")
            keywords.append(word)
            probabilities.append(int(frequency))

    # Normalize frequencies to create probabilities
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]
    
    return keywords, probabilities

# Function to sample keywords based on their probabilities and temperature
def sample_keywords(keywords, probabilities, k=5, temperature=1.0):
    """
    Sample 'k' keywords from the list using temperature-adjusted probabilities.

    Parameters:
    keywords (list): List of keywords to sample from.
    probabilities (list): List of probabilities for each keyword.
    k (int): Number of keywords to sample (default: 5).
    temperature (float): Temperature to adjust sampling distribution (default: 1.0).

    Returns:
    list: A list of 'k' sampled keywords.
    """
    # Adjust the probabilities based on temperature for more diverse sampling
    adjusted_probabilities = np.exp(np.log(probabilities) / temperature)
    adjusted_probabilities /= np.sum(adjusted_probabilities)
    
    # Sample 'k' keywords based on the adjusted probabilities
    return random.choices(keywords, adjusted_probabilities, k=k)

# Function to generate synthetic data for a given task
def generate_synthetic_data(num_samples=5, k=5, task_name="table_extraction", domain="AM", temperature=1.0):
    """
    Generate synthetic data by sampling keywords and generating task-specific prompts.

    Parameters:
    num_samples (int): Number of synthetic data samples to generate (default: 5).
    k (int): Number of keywords to sample for each synthetic data (default: 5).
    task_name (str): Name of the task for which to generate synthetic data. Options: 'table_extraction', 'multiple_choice', 'T_F', 'entity_extraction'.
    domain (str): Domain for keyword sampling (default: 'AM').
    temperature (float): Temperature to adjust sampling probabilities (default: 1.0).

    Returns:
    list: A list of synthetic data dictionaries, each containing system and user prompts.
    """
    # Select the appropriate template and examples based on the task_name
    if task_name == "table_extraction":
        template = table_extract_template
        examples = table_extract_example
    elif task_name == "multiple_choice":
        template = multiple_choice_template
        examples = multiple_choice_example
    elif task_name == "T_F":
        template = TF_template
        examples = TF_example  
    elif task_name == "entity_extraction":
        template = entity_extraction_template
        examples = entity_extraction_example
    else:
        raise ValueError("Unsupported task_name. Choose from 'table_extraction', 'multiple_choice', 'T_F', or 'entity_extraction'")

    # Load keywords and probabilities from the word frequency table for the specific domain
    keywords, keyword_probabilities = load_keywords(f'./parsed_reference/{domain}/word_frequency_table.txt')

    synthetic_data = []
    # Generate the specified number of synthetic data samples
    for i in range(num_samples):
        # Sample 'k' keywords from the keyword list
        sampled_keywords = ', '.join(sample_keywords(keywords, keyword_probabilities, k, temperature=temperature))

        # Generate system and user prompts using the selected template and sampled keywords
        system_prompt = template[0]["system"].format(examples=examples)
        user_prompt = template[0]["user"].format(keywords=sampled_keywords)
        
        # Append the generated synthetic data (system and user prompts) to the result list
        synthetic_data.append({
            "system": system_prompt,
            "user": user_prompt,
        })
    
    return synthetic_data
