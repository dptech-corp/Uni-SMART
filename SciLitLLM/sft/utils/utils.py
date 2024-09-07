import os
import json
import random
import time


# Function to save raw results to a file
def save_raw_results(results, output_path):
    """
    Saves raw results as a pickle file.

    Parameters:
    - results: The DataFrame containing results to be saved.
    - output_path: The location where the file will be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_pickle(output_path)


# Function to save synthetic data
def save_synthetic_data(results, domain, task, k, temperature, prev_data=[]):
    """
    Saves generated synthetic data as a JSON file.

    Parameters:
    - results: DataFrame containing the results.
    - domain: The domain for which synthetic data is generated.
    - task: The task type for synthetic data generation.
    - k: Number of keywords sampled.
    - temperature: Temperature for sampling.
    - prev_data: Previous data to which new samples will be appended (optional).
    """
    synthetic_data = prev_data.copy()
    for index, row in results.iterrows():
        result_content = row["result"]["choices"][0]["message"]["content"]

        try:
            parsed_content = json.loads(result_content, strict=False)

            structured_content = {
                "answer": str(parsed_content["answer"]),
                "text": str(parsed_content["text"])
            }
            
            synthetic_data.append(structured_content)
        except (json.JSONDecodeError, TypeError, Exception) as e:
            print(f"Error processing result {index}: {e}")
            continue
    
    num_samples = len(synthetic_data)
    output_path = f"results/{domain}/{task}_sample{num_samples}_k{k}_t{temperature}_synthetic_data.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(synthetic_data, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(synthetic_data)} synthetic data to {output_path}")


# Function to extract integer from filename
def extract_int_from_filename(filename):
    """
    Extracts an integer from the filename.

    Parameters:
    - filename: The filename from which to extract the integer.

    Returns:
    - The integer extracted from the filename or -1 if extraction fails.
    """
    try:
        return int(filename.split('_')[2][6:])
    except (IndexError, ValueError):
        return -1


# Function to generate a random seed
def generate_random_seed():
    """
    Generates a random seed based on current time.
    """
    random.seed(time.time())
