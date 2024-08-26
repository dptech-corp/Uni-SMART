## **Corpora Processing**

This sub-folder contains scripts for processing corpora, specifically focusing on parsing PDFs and correcting formatting and grammar errors introduced during the parsing process.

### PDF Parsing

Raw PDF files are located in the `resources/raw_pdf` directory. To parse all PDFs into text format, use the `pdf_parsing.py` script. The parsed texts will be saved in the `resources/parse-text.jsonl` file.

To run the PDF parsing script, use the following command:

```
python pdf_parsing.py --pdf_folder resources/raw_pdf --output_path resources/parse-text.jsonl

```

### Format and Grammar Correction

The PDF parsing process may introduce formatting and syntax errors. To correct these errors, we use the Llama3-8B model to enhance the text quality by addressing these issues.

#### Deploying the LLM

Before correcting formatting and grammar errors, you need to deploy a large language model (LLM) using [vllm](https://github.com/vllm-project/vllm) to increase throughput. Make sure to install `vllm` first.

Start the vllm server on `localhost:8000` by running the following command:

```
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --host localhost --port 8000
```

#### Running the Correction Script

Once the LLM is deployed, you can prompt it to correct the formatting and grammar errors introduced during the PDF parsing process.

To run the format and grammar correction script, use the following command:

```
python format_grammar_correction.py --input_path resources/parse-text.jsonl --output_path resources/parse-correction-text.jsonl --llm_endpoint localhost:8000 --model meta-llama/Meta-Llama-3-8B-Instruct --split_size 2048
```

This script will take the parsed text from `resources/parse-text.jsonl`, send it to the LLM for correction, and save the corrected textual splits (in words) in `resources/parse-correction-text.jsonl`.

### Quality Control

Directly performing quality control over our CPT corpora with pre-trained LLM could be costy. Instead, we use the LLM to label the quality score on a subset of the corpora, then use the labelled subset to train a BERT classifier to predict the quality score for the rest of the corpora.

#### Subset Labelling

Before training the BERT classifier, we need to label a subset of the corpora with quality scores. You can start a vllm server on `localhost:8000` by running the following command:

```bash
cd quality_control
bash start_vllm.sh
```

Then, you can use the following command to label the quality score on a subset of the corpora:

```bash
python llama_infer.py --input_file ../resources/parse-correction-text.jsonl --output_file ../resources/parse-correction-text-scored-50k.jsonl
```

#### Training the Quality Classifier

After labelling the subset of the corpora, you can train a BERT classifier to predict the quality score for the rest of the corpora. You can use the following command to train the classifier:

```bash
python train_edu_bert.py --no_eval --base_model_name="HuggingFaceFW/fineweb-edu-classifier" --dataset_path="../resources/parse-correction-text-scored-50k.jsonl" --target_column="prediction" --checkpoint_dir="checkpoints" --num_train_epochs 20 --per_device_train_batch_size 256 --learning_rate 0.001
```

#### Predicting the Quality Score

After training the classifier, you can use the following command to predict the quality score for the rest of the corpora:

```bash
bsah run_infer.sh
```

The predicted quality score will be saved in `../resources/parse-correction-text-scored.jsonl`, and the filtered top 75% corpora will be saved in `../resources/parse-correction-text-scored-top75.jsonl`.