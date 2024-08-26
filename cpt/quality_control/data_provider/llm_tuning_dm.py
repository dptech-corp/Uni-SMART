from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import json
import random

class TextbookDataset(Dataset):
    def __init__(self, data_path):
        super(TextbookDataset, self).__init__()
        with open(data_path, 'r') as f:
            lines = f.readlines()
            self.data_list = [json.loads(line.strip()) for line in lines]

    def shuffle(self):
        random.shuffle(self.data_list)
        return self

    def len(self):
        return len(self)
    
    def get(self, idx):
        return self.__getitem__(idx)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return index, self.data_list[index]['text'], raw_text

class InferenceCollater:
    def __init__(self, tokenizer, max_len):
        self.max_len = max_len
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        indices, input_text, raw_text = zip(*batch)
        ## deal with prompt
        self.tokenizer.padding_side = 'left'
        batch = self.tokenizer(text=input_text,
                                    truncation=True,
                                    padding='longest',
                                    add_special_tokens=True,
                                    max_length=self.max_len,
                                    return_tensors='pt',
                                    return_attention_mask=True)
        target_dict = {'input': raw_text, 'indices': indices}
        return batch, target_dict

class LLMInferringDM(LightningDataModule):
    def __init__(
        self,
        data_path,
        inference_batch_size,
        input_max_len,
        num_workers,
    ):
        super().__init__()
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = input_max_len
        self.test_dataset = TextbookDataset(data_path)
        self.tokenizer = None
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len),
        )
        return test_loader


