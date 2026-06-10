import os
import json
from torch.utils.data import Dataset

class TraceTaskDataset(Dataset):
    def __init__(self, data_root, task_name, split="train", max_examples=None):
        self.samples = []
        split_file = "eval.json" if split == "eval" else f"{split}.json"
        path = os.path.join(data_root, task_name, split_file)
        
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            
        if max_examples is not None and max_examples > 0:
            raw = raw[:max_examples]
            
        for item in raw:
            prompt = item.get("prompt", "").strip()
            answer = item.get("answer", "").strip()
            if prompt and answer:
                self.samples.append((prompt, answer))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class CollateFunction:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        prompts, answers = zip(*batch)
        full_texts = [f"{p}\n{a}{self.tokenizer.eos_token}" for p, a in zip(prompts, answers)]
        prompt_only = list(prompts)
        
        full_enc = self.tokenizer(
            full_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        
        prompt_enc = self.tokenizer(
            prompt_only, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        labels = input_ids.clone()
        
        for i, prompt_len in enumerate(prompt_enc["attention_mask"].sum(dim=1)):
            labels[i, :prompt_len] = -100
            
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels
        }