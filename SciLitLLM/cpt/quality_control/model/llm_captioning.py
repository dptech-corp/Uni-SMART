import os
import torch
import pytorch_lightning as pl
import json
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

import re
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

class LLMCaptioning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_new_tokens = args.max_new_tokens
        self.min_new_tokens = args.min_new_tokens
        self.llm_name = args.llm_name
        self.llm_job = args.llm_job
        self.result_path = args.result_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, padding_side='left')
        if 'classify' in self.llm_job:
            self.llm_model = AutoModelForSequenceClassification.from_pretrained(self.llm_name, torch_dtype=torch.bfloat16)
        elif self.llm_job == 'perplexity':
            self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
        elif self.llm_job == 'generate':
            self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
        
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.llm_model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.save_hyperparameters(args)

    def on_test_epoch_start(self) -> None:
        self.saved_dict_list = []

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        input_batch, target_dict = batch
        if self.llm_job == 'classify':
            predictions = self.classify(input_batch)
        if self.llm_job == 'classify_v2':
            predictions = self.classify_v2(input_batch)
        elif self.llm_job == 'perplexity':
            predictions = self.perplexity(input_batch)
        elif self.llm_job == 'generate':
            generated_text = self.generate(
                input_batch,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_new_tokens,
                min_length=self.min_new_tokens,
            )
            target_dict['generated_text'] = generated_text
            predictions = [parse_score(text) for text in generated_text]
        target_dict['prediction'] = predictions
        self.saved_dict_list.append(target_dict)
    
    def gather_dict_results(self, dict_list):
        list_of_dict_list = [None for _ in range(self.trainer.world_size)]
        dist.all_gather_object(list_of_dict_list, dict_list)
        dict_list = [i for ii in list_of_dict_list for i in ii] ## dict list, each dict has `val`ues that are lists of predictions, etc.
        keys = dict_list[0].keys()
        gathered_dict = {} # each value is a list of predictions, etc.
        for key in keys:
            gathered_dict[key] = [i for d in dict_list for i in d[key]]
        dict_list = []
        for i in range(len(gathered_dict['input'])):
            d = {k:gathered_dict[k][i] for k in keys}
            dict_list.append(d)
        return dict_list

    def save_results(self, dict_list):
        with open(self.result_path, 'w', encoding='utf8') as f:
            for dict_item in dict_list:
                f.write(json.dumps(dict_item, ensure_ascii=True) + '\n')
        top_75 = sorted(dict_list, key=lambda x: x['prediction'], reverse=True)[:int(len(dict_list)*0.75)]
        with open(self.result_path.replace('.jsonl', '-top75.jsonl'), 'w', encoding='utf8') as f:
            for dict_item in top_75:
                f.write(json.dumps(dict_item, ensure_ascii=True) + '\n')

    def on_test_epoch_end(self):
        result_list = self.gather_dict_results(self.saved_dict_list)
        self.saved_dict_list = []
        if self.global_rank == 0:
            self.save_results(result_list)

    @torch.no_grad()
    def classify(
        self,
        input_batch
    ):
        outputs = self.llm_model(
            input_ids=input_batch.input_ids,
            attention_mask=input_batch.attention_mask,
        )
        logits = outputs.logits.squeeze()
        probs = torch.nn.functional.softmax(logits, dim=1)
        probs = probs[:,1].squeeze()
        scores = probs.tolist()
        return scores

    @torch.no_grad()
    def classify_v2(
        self,
        input_batch
    ):
        outputs = self.llm_model(
            input_ids=input_batch.input_ids,
            attention_mask=input_batch.attention_mask,
        )
        logits = outputs.logits.squeeze()
        scores = logits.tolist()
        return scores
    
    @torch.no_grad()
    def perplexity(
        self,
        input_batch
    ):
        pass

    @torch.no_grad()
    def generate(
        self, 
        input_batch,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1
        ):
        seq_len = input_batch.input_ids.shape[1]
        outputs = self.llm_model.generate(
            input_ids=input_batch.input_ids,
            attention_mask=input_batch.attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=seq_len+max_length,
            min_length=seq_len+min_length,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        outputs = [output[seq_len:] for output in outputs]
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        return output_text

