# Synthetic Instruction Generation for Targeted Scientific Fields

This project offers a pipeline for generating diverse, high-quality instructions tailored to specific scientific domains, along with their corresponding question-answer pairs. The pipeline comprises four key steps:

1. **Step 1: Domain Keyword Probability Table Creation**
2. **Step 2: Scientific Task Description Collection**
3. **Step 3: Instruction Generation**
4. **Step 4: Quality Filtering**

---

## Step 1: Domain Keyword Probability Table Creation

For each domain (e.g., alloys, biomedicine, materials), collect relevant documents and store them in `/reference_papers/<domain_name>`. To build a probability table of domain-specific keywords, execute the following script:

```bash
python helper/parse_pdfs.py
```

This will parse the PDFs and generate `word_frequency_table.txt` for each domain, capturing word-level distributions from domain literature.

---

## Step 2: Scientific Task Description Collection

We provide predefined scientific tasks, including:

1. Table Extraction
2. Entity Extraction
3. Multiple Choice Questions
4. True-or-False Questions
5. Molecule Translation
6. Molecule Extraction

These tasks leverage domain-specific keywords for instruction diversity. Molecule Translation and Molecule Extraction tasks, in particular, use a list of molecular formulas to guide the process.

---

## Step 3: Instruction Generation

To generate instructions for a specific domain, use:

```bash
python gen_synthetic_data.py --k <number_of_keywords> --num_samples <number_of_samples> --domain <domain_name> --task <task_name> --save_results --temperature <sampling_temperature>
```

### Parameters:

- `k`: Number of keywords to sample (default: 20)
- `num_samples`: Number of samples to generate
- `domain`: The domain for which instructions are being generated
- `task`: Task type (choose from `'table_extraction', 'multiple_choice', 'T_F', 'entity_extraction', 'molGen_wocontext', 'molGen_wcontext'`)
- `temperature`: Sampling temperature (default: 3)

Make sure to configure your Azure endpoints and API keys in `gen_synthetic_data.py`. Then, as an example, you can run:

```bash
python gen_synthetic_data.py --k 20 --num_samples 1 --domain example_domain --task table_extraction --save_results --temperature 3
```

### Results

The generated instructions and question-answer pairs will be saved in the `./results/` folder, following this naming convention:

```bash
results/{domain}/{task}_sample{num_samples}_k{k}_t{temperature}_synthetic_data.json
```

Example content:

```json
[
    {
        "answer": "b) James Wakasa",
        "text": "James Wakasa, an eminent researcher at Brightmanfound, recently published a comprehensive report on the advancements in nanocrystal applications..."
    }
]
```

---

## Step 4: Quality Filtering

We implement two main methods for quality control: deduplication and LLM-based filtering.

To run deduplication:

```bash
python helper/dedup.py
```

After deduplication, you can score the generated instructions using the LLM-based filtering method:

```bash
python helper/infer_sft.py --input_file "deduped_results" --output_file "sft-data-scored.json" 
```

---

## Customizing Tasks

To add new tasks, create a prompt template and few-shot examples similar to `prompts/table_extract_prompt.py`. Then, implement task-specific generation functions like those in `utils/mc_tf_table_entity_gen.py`. Finally, update `gen_synthetic_data.py` to support the new task.

---
