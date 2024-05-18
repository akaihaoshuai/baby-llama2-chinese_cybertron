import os
import torch
from tqdm import tqdm
import jsonlines
import random
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from src.utils import get_ctx

from src.benchmark.metrics import qa_f1_score

class GSM8K:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_path_name,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path_name = model_path_name

    def run(self, data_path, benchmark_dir, shot: int):
        results, accs = {}, {}

        dir_name = os.path.join(benchmark_dir, 'GSM8K')
        os.makedirs(dir_name, exist_ok=True)

        train_data = []
        with open(os.path.join(data_path, "train.jsonl"), "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                train_data.append(item)
        test_data = []
        with open(os.path.join(data_path, "test.jsonl"), "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                test_data.append(item)

        device = next(self.model.parameters()).device
        ctx = get_ctx(device)

        acc = 0.0
        for data in tqdm(test_data):
            prompt = ''
            if shot != 0:
                if isinstance(train_data, list):
                    random.shuffle(train_data)
                    shuffled = train_data
                else:
                    shuffled = train_data.shuffle()
                for i in range(min(shot, len(shuffled))):
                    prompt += "\n" + self.build_example(shuffled[i], with_answer=True)
            prompt += "\n" + self.build_example(data, with_answer=False)

            x=self.tokenizer.encode(prompt, add_special_tokens=False)+[self.tokenizer.special_tokens['<eos>']]
            x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
            with ctx:
                y = self.model.generate(x)[0]

            pred = self.tokenizer.decode(y, skip_special_tokens=True)
            acc += qa_f1_score(pred, data["answer"])
            
        acc /= len(test_data)
        return results, acc
    
    def build_example(self, data, with_answer: bool = True):
        question = data["question"]
        answer = data["answer"].strip().upper() if with_answer else ""
        return f"{question}\n\nanswer: {answer}"
    