import os
import asyncio
import argparse
import pandas as pd
import json

from utils.api_gpt import Batch, create_azure_config
from utils.mol_gen_wo_context import generate_synthetic_data as generate_mol_wo_context_data
from utils.mc_tf_table_entity_gen import generate_synthetic_data
from utils.mol_gen_w_context import generate_synthetic_data as generate_mol_w_context_data
from utils.utils import save_synthetic_data, extract_int_from_filename, generate_random_seed

# Function to run the batch processing for GPT completions
async def run_batch(batch, synthetic_data, model="gpt-35-turbo", tpm=10000):
    """
    Runs batch processing for GPT completions.

    Parameters:
    - batch: The batch object to which data is added.
    - synthetic_data: List of synthetic data to be processed.
    - model: GPT model to be used (default: "gpt-35-turbo").
    - tpm: Tokens per minute limit (default: 10000).
    """
    for data in synthetic_data:
        await batch.add(
            "chat.completions.create",
            model=model,
            messages=[{"role": "system", "content": data["system"]},
                      {"role": "user", "content": data["user"]}]
        )
    
    return await batch.run()


# Main function to run the pipeline
async def main(num_samples, model, k, domain, task, save_results=False, temperature=1, print_key_only=False):
    generate_random_seed()

    # Check for previous runs with required or more samples
    if not os.path.exists(f"results/{domain}"):
        os.makedirs(f"results/{domain}")
    previous_files = [f for f in os.listdir(f"results/{domain}") if f.startswith(f"{task}_sample") and f.endswith(f"_k{k}_t{temperature}_synthetic_data.json")]
    previous_files_sorted = sorted(previous_files, key=extract_int_from_filename, reverse=True)
    
    prev_results = []
    already_processed_samples = 0

    if previous_files_sorted:
        file = previous_files_sorted[0]
        previous_num_samples = int(file.split('_')[2][6:])
        if previous_num_samples >= num_samples:
            print(f"Synthetic data already exists for {task} with {previous_num_samples} samples and {k} keywords, temperature {temperature}.")
            return
        else:
            print(f"Loading synthetic data from previous run with {previous_num_samples} samples.")
            with open(f"results/{domain}/{file}", 'r') as f:
                prev_results = json.load(f)
            already_processed_samples = previous_num_samples

    samples_to_run = num_samples - already_processed_samples

    if task == 'table_extraction':
        samples_to_run = int(3 * samples_to_run)
    else:
        samples_to_run = int(1.5 * samples_to_run)
    
    print(f"Generating synthetic data for {samples_to_run} samples.")

    # Generate synthetic data for the full num_samples
    if task in ['table_extraction', 'multiple_choice', 'T_F', 'entity_extraction']:
        synthetic_prmopt = generate_synthetic_data(num_samples=samples_to_run,
                                                   k=k,
                                                   task_name=task,
                                                   domain=domain,
                                                   temperature=temperature)
    elif task == 'molGen_wocontext':
        synthetic_prmopt = generate_mol_wo_context_data(num_samples=samples_to_run)
    elif task == 'molGen_wcontext':
        synthetic_prmopt = generate_mol_w_context_data(num_samples=samples_to_run)
    else:
        raise ValueError(f"Invalid task: {task}. Only 'table_extraction' and 'mol_gen' are supported.")
           
    azure_configs = []
    if azure_configs == []:
        raise ValueError("You need to set up your own Azure configs for the GPT models.")

    # TODO: remove below for the final version!!!
    # azure_configs = [
    #     {"azure_endpoint": "https://shiyigong.openai.azure.com/", "api_version": "2024-03-01-preview", "api_key": "51cd69e0e8a14ab9a01332815f6af7d9"},
    #     {"azure_endpoint": "https://zhaozhongyao.openai.azure.com/", "api_version": "2024-03-01-preview", "api_key": "57ee17fbb7bf4f0b82c60d584a67f6f1"},
    #     {"azure_endpoint": "https://wangganchang.openai.azure.com/", "api_version": "2024-03-01-preview", "api_key": "20930ea10e184d1aa8d743fba77eca77"},
    # ]

    num_batches = len(azure_configs)
    subset_size = len(synthetic_prmopt) // num_batches
    subsets = [synthetic_prmopt[i * subset_size:(i + 1) * subset_size] for i in range(num_batches)]
    subsets[-1].extend(synthetic_prmopt[num_batches * subset_size:])

    tpm = 50000
    tasks_list = []

    for config in azure_configs:
        azure_config = create_azure_config(config)
        azure_config.display_endpoint()
        batch = Batch(tpm=tpm, azure=azure_config)
        task_now = run_batch(batch, subsets.pop(), model=model)
        tasks_list.append(task_now)

    new_results = await asyncio.gather(*tasks_list)

    combined_results = pd.concat(new_results, ignore_index=True)

    if save_results:
        save_synthetic_data(combined_results, domain, task, k, temperature, prev_data=prev_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data using GPT models.")
    parser.add_argument('--num_samples', type=int, default=2, help='Number of samples to generate')
    parser.add_argument('--model', type=str, default="gpt-4o", help='GPT model to use')
    parser.add_argument('--k', type=int, default=20, help='Number of keywords to sample')
    parser.add_argument('--domain', type=str, default='AM', help='Domain for instruction generation')
    parser.add_argument('--task', type=str, default='table_extraction', help='Task to specify the type of synthetic data generation')
    parser.add_argument('--save_results', action='store_true', help='Save raw results and synthetic data')
    parser.add_argument('--temperature', type=float, default=3.0, help='Temperature for sampling keywords')
    parser.add_argument('--print_key_only', action='store_true', help='Print only the keywords')
    parser.add_argument('--force_run', action='store_true', help='Force run the synthetic data generation (overwrite existing)')

    args = parser.parse_args()

    if 'molGen' in args.task:
        args.domain = 'DC'

    models = ["gpt-35-turbo-16k", "gpt-4-128k", "gpt-4o"]
    if args.model not in models:
        raise ValueError(f"Invalid model: {args.model}. Choose from {models}")

    asyncio.run(main(args.num_samples,
                     args.model,
                     args.k,
                     args.domain,
                     args.task,
                     args.save_results,
                     args.temperature,
                     args.print_key_only))
