import openai
from openai import OpenAI
import json
import argparse
import re
import os

def write_json(data, filename):
    print(f'Writing {len(data)} data to {filename}...')
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f'Reading {len(data)} data from {filename}...')
    return data

MY_SYSTEM_PROMPT = \
"""You are a helpful and precise assistant for checking the quality of instruction tuning data for large language models. Your task is to evaluate the given instruction using the criterions described below.

- Clarity: The sample should be clear, specific, and unambiguous, providing a well-defined task for the model to perform.
- Complexity: The sample should be advanced complexity that necessitate a high level of comprehension and cognitive processing, challenging the language model significantly.
- Correctness: The sample is impeccably written, with flawless grammar, syntax, and structure, demonstrating exceptional clarity and professionalism.
- Usefulness: The sample should be highly useful, and contribute to expanding the model's knowledge base.
- Adaptability: The sample could be adapted to different contexts or use cases, showing some flexibility.

After examining the instruction-response pair:
- Briefly justify your scores with a paragraph in the field "Explanation", up to 500 words.
- For each point of criterion above, assign a score from 1 to 5.
- You should only provide the rest of your answer in a structured format as shown below, and make sure your response can be directly parsed by computer programs.

Below is a template for your response:

Explanation: <string, your explanations to the scores>
====================
{
    "Clarity": <int, complexity_score>,
    "Complexity": <int, complexity_score>,
    "Correctness": <int, quality_score>,
    "Usefulness": <int, usefulness_score>,
    "Adaptability": <int, adaptability_score>
    "Total": <int, total_score>
}
"""

MY_USER_PROMPT = \
"""[Instruction Question]
{QUESTION}

[Your Score]
"""


def parse_json_score(text):
    pattern = r'\{[^{}]*\}'
    match = re.search(pattern, text)
    try:
        assert match
        content = match.group()
        result = json.loads(content)
        if isinstance(result, dict):
            return result
        else:
            return None
    except (SyntaxError, ValueError):
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser")

    parser.add_argument('--name', default='none', type=str)
    parser.add_argument('--input_file', default='none', type=str)
    parser.add_argument('--output_file', default='none', type=str)
    parser.add_argument('--base_url', default='http://127.0.0.1:28589/v1', type=str)
    parser.add_argument('--llm_model', default='Meta-Llama-3-70B-Instruct', type=str)
    args = parser.parse_args()
    return args

def get_qa_pair(data_item):
    question = [d for d in data_item['conversations'] if d['from'] == 'human'][0]['value']
    answer = [d for d in data_item['conversations'] if d['from'] == 'gpt'][0]['value']
    return question, answer

class LLamaUser:
    def __init__(self, input_file, output_file, base_url, llm_model):
        self.input_file = input_file
        self.output_file = output_file
        self.client = OpenAI(
            api_key = "EMPTY",
            base_url = base_url
        )
        self.model = llm_model

    def predict(self, data_item):
        try:
            question, answer = get_qa_pair(data_item)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": MY_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": MY_USER_PROMPT.replace("{QUESTION}", question).replace("{ANSWER}", answer)
                    }
                ],
            )
        except openai.BadRequestError:
            data_item['text'] = data_item['text'][:int(len(data_item['text'])/2)]
            # half the len of input text
            return self.predict(data_item)

        data_item['response'] = completion.choices[0].message.content
        try:
            data_item['prediction'] = parse_json_score(data_item['response'])
        except:
            data_item['prediction'] = None
        return data_item

    def run(self):
        data = []
        domains =  [d for d in os.listdir(self.input_file) if os.path.isdir(os.path.join(self.input_file, d))]
        for task in domains:
            for filename in os.listdir(os.path.join(self.input_file, task)):
                data_path = os.path.join(self.input_file, task, filename)
                data.extend(read_json(data_path))
        results = [self.predict(d) for d in data]
        result_dict = {}
        print(f"Results (total {len(results)}):")
        for d in results:
            if d['prediction'] and 'Total' in d['prediction']:
                score = d['prediction']['Total']
                if score not in result_dict:
                    result_dict[score] = 0
                result_dict[score] += 1
        for key in sorted(result_dict.keys()):
            print(f"{key}: {result_dict[key]}")
        write_json(results, self.output_file)
        
        filtered_data = [d for d in results if d['prediction'] and 'Total' in d['prediction'] and d['prediction']['Total'] >= 20]
        write_json(filtered_data, self.output_file.replace('.json', '-filtered.json'))

def main(args):
    user = LLamaUser(
        input_file = args.input_file,
        output_file = args.output_file,
        base_url = args.base_url,
        llm_model = args.llm_model,
    )
    user.run()

if __name__=='__main__':

    main(args=parse_args())
