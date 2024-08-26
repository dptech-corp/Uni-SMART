import os
import argparse
import json
import concurrent.futures
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm

# Configure OpenAI client
client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="http://localhost:8000/v1"
)

template = """I have extracted the following raw text from a PDF, but the extraction process has introduced many formatting issues such as unnecessary line breaks, extra spaces, and other artifacts that disrupt the text flow. Could you please help me correct these formatting issues and provide a clean, readable version of the text? Respond with the Corrected Version only.

Raw Text:

{RawText}

Start your response with "Here is the corrected version of the text:".
"""

def chunk_text(text, chunk_size):
    """Splits the text into chunks based on the chunk size (number of words)."""
    words = text.split()
    # Create chunks of text with chunk_size words, joining words back into a single string
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def process_chunk(chunk, model):
    """Processes a single chunk of text using LLM."""
    messages = [{"role": "user", "content": template.replace("{RawText}", chunk)}]
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        corrected_chunk = completion.choices[0].message.content.strip().replace(
            "Here is the corrected version of the text:", ""
        ).strip()
        return corrected_chunk
    except Exception as e:
        print(f"Error processing text chunk: {e}")
        return chunk

def process_text_entry(text, model, chunk_size, max_workers):
    """Processes a single text entry by splitting it into character-based chunks and using LLM to correct it."""
    corrected_text = []
    chunks = chunk_text(text, chunk_size)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(process_chunk, chunk, model): chunk for chunk in chunks}

        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                corrected_chunk = future.result()
                corrected_text.append(corrected_chunk)
            except Exception as e:
                print(f"Error processing chunk: {e}")
                corrected_text.append(future_to_chunk[future])

    return " ".join(corrected_text)

def split_text_to_entries(text, split_size):
    """Splits the text into smaller chunks based on paragraph boundaries and word count limit."""
    paragraphs = text.split('\n\n')  # Assuming paragraphs are separated by double newline
    entries = []
    split_id = 1
    current_entry = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        # Check if adding this paragraph would exceed the split_size
        if current_word_count + paragraph_word_count > split_size:
            # Save the current entry and reset
            entries.append({
                'text': "\n\n".join(current_entry),
                'meta_data': {'split_id': split_id}
            })
            split_id += 1
            current_entry = []
            current_word_count = 0
        
        current_entry.append(paragraph)
        current_word_count += paragraph_word_count
    
    # Add the last entry if it has any content
    if current_entry:
        entries.append({
            'text': "\n\n".join(current_entry),
            'meta_data': {'split_id': split_id}
        })

    return entries


def process_jsonl(input_path, output_path, model, chunk_size, split_size, max_workers):
    """Reads from input JSONL, processes text fields, and writes to output JSONL."""
    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()
        
        for line in tqdm(lines, desc="Processing JSONL"):
            try:
                entry = json.loads(line)
                text = entry['text']
                corrected_text = process_text_entry(text, model, chunk_size, max_workers)
                
                split_entries = split_text_to_entries(corrected_text, split_size)
                for split_entry in split_entries:
                    json.dump(split_entry, outfile)
                    outfile.write('\n')
                
                processed_count += 1
            except Exception as e:
                print(f"Error processing entry: {e}")
    
    print(f"Total processed entries: {processed_count}")

def main():
    parser = argparse.ArgumentParser(description="Correct formatting and grammar of text fields in a JSONL file using an LLM.")
    parser.add_argument('--input_path', type=str, default='resources/parse-text.jsonl', help="Path to the input JSONL file.")
    parser.add_argument('--output_path', type=str, default='resources/parse-correction-text.jsonl', help="Path to the output JSONL file.")
    parser.add_argument('--chunk_size', type=int, default=1024, help="Maximum size of text chunks to process with LLM.")
    parser.add_argument('--split_size', type=int, default=2048, help="Maximum characters of splited texts.")
    parser.add_argument('--max_workers', type=int, default=2, help="Maximum number of concurrent workers.")
    parser.add_argument('--llm_endpoint', type=str, default='localhost:8000', help="Base URL for the LLM API.")
    parser.add_argument('--model', type=str, default='Meta-Llama-3-8B-Instruct', help="Model name to use for the LLM.")

    args = parser.parse_args()
    
    # Configure OpenAI client with base URL and API key
    client.base_url = args.llm_endpoint

    process_jsonl(args.input_path, args.output_path, args.model, args.chunk_size, args.split_size, args.max_workers)

if __name__ == "__main__":
    main()
