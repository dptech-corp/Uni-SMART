import os
import PyPDF2
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

# Download stop words list
nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyPDF2.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def save_text_to_file(text, output_path):
    """
    Save the extracted text to a file.
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)

def clean_text(text):
    """
    Clean the extracted text by removing non-alphanumeric characters and converting to lowercase.
    Also, filter out numbers.
    """
    text = re.sub(r'\W+', ' ', text)  # Replace non-alphanumeric characters with spaces
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()
    return text

def load_stop_words():
    """
    Load stop words from the NLTK library.
    """
    return set(stopwords.words('english'))

def get_chemical_elements():
    """
    Return a set of chemical elements.
    """
    return {
        'h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne', 'na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar',
        'k', 'ca', 'sc', 'ti', 'v', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br',
        'kr', 'rb', 'sr', 'y', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te',
        'i', 'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm',
        'yb', 'lu', 'hf', 'ta', 'w', 're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
        'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu', 'am', 'cm', 'bk', 'cf', 'es', 'fm', 'md', 'no', 'lr',
        'rf', 'db', 'sg', 'bh', 'hs', 'mt', 'ds', 'rg', 'cn', 'uut', 'fl', 'uup', 'lv', 'uus', 'uuo'
    }

def construct_word_frequency_table(folder_path, parsed_folder_path):
    """
    Construct a word frequency table from all PDF files in a given folder, save parsed text, and filter out stop words and numbers.
    """
    word_counter = Counter()
    stop_words = load_stop_words()
    chemical_elements = get_chemical_elements()

    if not os.path.exists(parsed_folder_path):
        os.makedirs(parsed_folder_path)

    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                output_txt_path = os.path.join(parsed_folder_path, os.path.splitext(filename)[0] + '.txt')
                save_text_to_file(text, output_txt_path)

                cleaned_text = clean_text(text)
                words = [word for word in cleaned_text.split() 
                         if word not in stop_words 
                         and (len(word) > 2 or word in chemical_elements)]
                word_counter.update(words)

    return word_counter

def save_word_frequency_table(word_counter, output_path):
    """
    Save the word frequency table to a file.
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("Word Frequency Table:\n")
        for word, count in word_counter.most_common():
            file.write(f"{word}: {count}\n")

# list all domain directories under the foler
folder = "reference_papers"
domains = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

for domain in domains:
    folder_path = f"./reference_papers/{domain}"
    parsed_folder_path = f'./parsed_reference/{domain}'
    output_word_table_path = f'./parsed_reference/{domain}/word_frequency_table.txt'

    word_counter = construct_word_frequency_table(folder_path, parsed_folder_path)
    save_word_frequency_table(word_counter, output_word_table_path)

