import tqdm
import openai
from openai import OpenAI
import random
import json
import re
import argparse

SYSTEM_PROMPT = \
"""
Below is an extract from a textbook. Evaluate whether the text has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students. 
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

After examining the extract: 
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Educational score:  <total points>"
"""

USER_PROMPT = \
"""
The extract:
{}

Your answer:
"""


def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        lines = [json.dumps(d)+'\n' for d in data]
        f.writelines(lines)

def read_jsonl(filename):
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def parse_score(text):
    pattern = re.compile(r'Education(?:al)? score:\s*(\d+(?:\.\d+)?)(?:\/\d+)?')
    match = pattern.search(text)

    if match:
        score_str = match.group(1)
        if '.' in score_str:
            return int(float(score_str))
        else:
            return int(score_str)

    return -1

class LLamaUser:
    def __init__(self, input_file, output_file, base_url, llm_model, subset_num=50000):
        self.input_file = input_file
        self.output_file = output_file
        self.client = OpenAI(
            api_key = "EMPTY",
            base_url = base_url
        )
        self.model = llm_model
        self.subset_num = subset_num
    
    def predict(self, data_item):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": USER_PROMPT.format(data_item['text'])
                    }
                ],
            )
        except openai.BadRequestError:
            data_item['text'] = data_item['text'][:int(len(data_item['text'])/2)]
            # half the len of input text
            return self.predict(data_item)
            
        data_item['response'] = completion.choices[0].message.content
        data_item['prediction'] = parse_score(data_item['response'])
        return data_item
    
    def run(self):
        data = read_jsonl(self.input_file)
        if self.subset_num > 0:
            data = random.sample(data, self.subset_num)
        for data_item in tqdm.tqdm(data):
            data_item = self.predict(data_item)
        write_jsonl(data, self.output_file)

def main(args):
    user = LLamaUser(
        input_file = args.input_file,
        output_file = args.output_file,
        base_url = args.base_url,
        llm_model = args.llm_model,
    )
    user.run()

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser")

    parser.add_argument('--name', default='none', type=str)
    parser.add_argument('--input_file', default='none', type=str)
    parser.add_argument('--output_file', default='none', type=str)
    parser.add_argument('--base_url', default='http://localhost:8000/v1', type=str)
    parser.add_argument('--llm_model', default='Meta-Llama-3-70B-Instruct', type=str)
    args = parser.parse_args()
    return args

main(args=parse_args())