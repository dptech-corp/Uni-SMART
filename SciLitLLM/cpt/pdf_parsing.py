import os
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from PyPDF2 import PdfReader
from tqdm import tqdm

def process_pdf(pdf_path):
    """Extract text from a PDF and return a tuple of the file name, success status, and text content."""
    file_name = os.path.basename(pdf_path)
    try:
        reader = PdfReader(pdf_path)
        text_content = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
        return (file_name, True, "\n".join(text_content))
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return (file_name, False, None)

def main(pdf_folder, output_path):
    """Main function to process all PDFs in a folder and save the output to a JSONL file."""
    pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.lower().endswith('.pdf')]

    processed_files = []
    cpu_count = os.cpu_count()
    with ProcessPoolExecutor(max_workers=int(0.9 * cpu_count)) as executor:
        for result in tqdm(executor.map(process_pdf, pdf_files), total=len(pdf_files), desc="Processing PDFs"):
            processed_files.append(result)

    # Write the results to a JSONL file
    with open(output_path, 'w', encoding='utf-8') as jsonl_file:
        for file_name, success, text_content in processed_files:
            if success:
                jsonl_file.write(json.dumps({'text': text_content, 'meta_data': {'source': file_name}}) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF files and extract text.")
    parser.add_argument('--pdf_folder', type=str, default='resources/raw_pdf', help="Path to the folder containing PDF files.")
    parser.add_argument('--output_path', type=str, default='resources/parse-text.jsonl', help="Path to the JSONL output file.")

    args = parser.parse_args()
    main(args.pdf_folder, args.output_path)
