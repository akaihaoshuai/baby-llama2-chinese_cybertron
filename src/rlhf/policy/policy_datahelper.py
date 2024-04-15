import os
import random
import logging
import torch
import json
import copy
from typing import List, Dict, Any, Tuple
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from torch.utils.data import get_worker_info, IterableDataset
from src.rlhf.utils import pad_sequences
import langid

def get_human_prompt(language):
    return "<|Human|>" if language == 'zh' else "Human:"


def get_assistant_prompt(language):
    return "<|MOSS|>" if language == 'zh' else "Assistant:"


def get_special_prompt(i, language):
    return get_human_prompt(language) if i % 2 == 0 else get_assistant_prompt(language)

def get_model_prompt(context: List[str], eos_token="</s>", opt=None):
    language_predicted = langid.classify(context[0])[0].strip()  
    human_prompt, assistant_prompt = get_human_prompt(language_predicted), get_assistant_prompt(language_predicted)
    if context[-1].startswith(human_prompt):
        end_prompt = assistant_prompt
    elif context[-1].startswith(assistant_prompt):
        end_prompt = human_prompt
    else:
        raise ValueError
        
    context = eos_token.join(context)
    return f'{context}{eos_token}{end_prompt}'

class IterDataset(IterableDataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return self.size
    
    def sample_generator(self):
        random.seed(None)
        
        worker_info = get_worker_info()
        if worker_info is not None:
            self.data = self.data[worker_info.id::worker_info.num_workers]
            # logging.info(f'Worker {worker_info.id}: {len(self.data)} samples.')
            
        if self.mode == 'train':
            random.shuffle(self.data)

        for sample in self.data:
            yield self.format(sample)

    def batch_generator(self):
        batch = []

        for sample in self.sample_generator():
            sample_len = len(sample['text_vec'])
            if sample_len > self.opt.max_seq_len:
                logging.warn(f'Get sample length: {sample_len} > {self.opt.max_seq_len}.')
                continue

            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch

    def final_generator(self):
        data_generator = self.batch_generator()
        for batch_samples in data_generator:
            batch = self.batchify(batch_samples)
            yield batch

    def __iter__(self):
        return self.final_generator()
    
            
class OnlyPromptDataset(IterDataset):
    def __init__(self, opt, tokenizer, accelerator, mode = 'train', **kwargs) -> None:
        super().__init__()
        self.opt = opt
        self.mode = mode
        self.accelerator = accelerator
        self.tokenizer = tokenizer

        self.data = []
        files = sorted([file for file in os.listdir(opt.ppo_data_path) if file.endswith(f'{mode}.json')])
        for file in files:
            file_path = os.path.join(opt.ppo_data_path, file)
            tmp_data = []
            try:
                tmp_data = self.load_data(file_path)
            except Exception as e:
                logging.warn(f"Loading samples from {file_path} failed. {str(e)}...")
            self.data.extend(tmp_data)
            logging.info(f'Loaded {len(tmp_data)} samples from {file_path}.')
        logging.info(f'=============Loaded total {len(self.data)} samples from {files}.=============')

        self.size = len(self.data)

        if accelerator and self.accelerator.use_distributed:
            self.data = self.data[self.accelerator.process_index::self.accelerator.num_processes]

        self.batch_size = opt.num_rollouts # batch size for sampling from env
    
    def load_data(self, file_path: str):
        with open(file_path, 'r') as f:
            data: List[List[str]] = json.load(f)
            
        output: List[List[str]] = [sample for sample in data if all(sample)]
        del data

        return output
    
    def format(self, sample: List[str]) -> Dict[str, Any]:
        context = sample
        language_predicted = langid.classify(context[0])[0].strip()  
        context = [get_special_prompt(i + (len(context) + 1) % 2, language_predicted) + s for i, s in enumerate(context)]
        context_vec = self.tokenizer.encode(get_model_prompt(context, self.tokenizer.eos_token, self.opt), add_special_tokens=True)
        
        # truncate to max_len
        while len(context_vec) > self.opt.max_seq_len and len(context) > 1:
            context = context[1:]
            context_vec = self.tokenizer.encode(get_model_prompt(context, self.tokenizer.eos_token, self.opt), add_special_tokens=True)
            
        output = {
            'text': self.tokenizer.decode(context_vec, skip_special_tokens=False),
            'text_vec': context_vec
        }
    
        return output

    # batchify for single format(sample)
    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_text_vec = torch.tensor(pad_sequences(
            [sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.pad_token_id, padding='left'
            ), dtype=torch.long)
        return {
            'text_vec': batch_text_vec,
            'text': [sample['text'] for sample in batch_samples]
        }

    def batch_generator(self):
        while True:
            for batch in super().batch_generator():
                if len(batch) == self.batch_size:
                    yield batch
            if self.mode != 'train':
                break


class ExperienceDataset(IterDataset):
    def __init__(self, data, opt, tokenizer, accelerator, mode = 'train', **kwargs) -> None:
        self.opt = opt
        self.mode = mode
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        
        self.use_ppo_pretrain_loss = opt.use_ppo_pretrain_loss
        self.batch_size = opt.batch_size
        self.gamma = opt.gamma
        self.lam = opt.lam
        self.data = data
        self.size = len(data)

        if self.accelerator.use_distributed:
            self.size *= self.accelerator.num_processes

    def get_advantages_and_returns(self, rewards: List[float], values: List[float]):
        '''
        Copied from TRLX: https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        '''
        response_length = len(values)
        advantages_reversed = []
        lastgaelam = 0
        for t in reversed(range(response_length)):
            nextvalues = values[t + 1] if t < response_length - 1 else 0.0
            delta = rewards[t] + self.gamma * nextvalues - values[t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            
        advantages = advantages_reversed[::-1]
        returns = [a + v for a, v in zip(advantages, values)]
        assert len(returns) == len(advantages) == len(values)
        return advantages, returns
    
    def format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        output = copy.deepcopy(sample)
        advantages, returns = self.get_advantages_and_returns(sample['reward'], sample['values'])
        context_vec, resp_vec = sample['context_vec'], sample['resp_vec']
        assert len(resp_vec) == len(advantages) == len(returns)
        
        text_vec = context_vec + resp_vec
        loss_mask = [0] * len(context_vec) + [1] * len(resp_vec)

        output['text'] = self.tokenizer.decode(text_vec, skip_special_tokens=False)
        output['text_vec'] = text_vec
        output['res_len'] = len(resp_vec)
        output['logprobs'] = [0.] * (len(context_vec) - 1) + output['logprobs']
        output['loss_mask'] = loss_mask
        
        output['reward'] = sample['reward']
        output['values'] = [0.] * (len(context_vec) - 1) + output['values']
        output['advantages'] = [0.] * (len(context_vec) - 1) + advantages
        output['returns'] = [0.] * (len(context_vec) - 1) + returns

        return output
    
    def batch_generator(self):
        for batch in super().batch_generator():
            yield batch

    # batchify for single format(sample)   
    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            'text': [sample['text'] for sample in batch_samples],
            'text_vec': torch.tensor(pad_sequences([sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.pad_token_id), dtype=torch.long),
            'res_len': [sample['res_len'] for sample in batch_samples],
            'logprobs': torch.tensor(pad_sequences([sample['logprobs'] for sample in batch_samples], pad_value=0.)),
            'loss_mask': torch.tensor(pad_sequences([sample['loss_mask'] for sample in batch_samples], pad_value=0), dtype=torch.bool),
            'ppl_value': torch.tensor([sample['ppl_value'] for sample in batch_samples]),
            'ppl0_value': torch.tensor([sample['ppl0_value'] for sample in batch_samples]),
            
            'reward': [sample['reward'] for sample in batch_samples],
            'values': torch.tensor(pad_sequences([sample['values'] for sample in batch_samples], pad_value=0.)),
            'advantages': torch.tensor(pad_sequences([sample['advantages'] for sample in batch_samples], pad_value=0.)),
            'returns': torch.tensor(pad_sequences([sample['returns'] for sample in batch_samples], pad_value=0.))
        }

        if self.use_ppo_pretrain_loss:
            tmp_ppo_context_vec = []
            for pretrain_data_batch in [sample['ppo_context_vec'] for sample in batch_samples]:
                for one_sample in pretrain_data_batch:
                    tmp_ppo_context_vec.append(one_sample)

            batch['ppo_context_vec'] = torch.tensor(pad_sequences(
                tmp_ppo_context_vec, pad_value=self.tokenizer.pad_token_id
                ), dtype=torch.long)
            del tmp_ppo_context_vec

            tmp_ppo_loss_mask = []
            for pretrain_data_batch in [sample['ppo_loss_mask'] for sample in batch_samples]:
                for one_sample in pretrain_data_batch:
                    tmp_ppo_loss_mask.append(one_sample)
            batch['ppo_loss_mask'] = torch.tensor(pad_sequences(tmp_ppo_loss_mask, pad_value=0), dtype=torch.bool)
            del tmp_ppo_loss_mask

        return batch


class PPOSFTDataset(IterDataset):
    def __init__(self, opt, tokenizer, accelerator, **kwargs):
        self.opt = opt
        self.mode = 'train'
        self.accelerator = accelerator
            
        self.tokenizer = tokenizer
        self.batch_size = opt.ppo_pretrain_batch_size_ratio

        self.data = []
        for file in os.listdir(opt.ppo_pretrain_data_path):
            if file.endswith(f'{self.mode}.json'):
                file_path = os.path.join(opt.ppo_pretrain_data_path, file)
                tmp_data = []
                tmp_data = self.load_data(file_path)
          
                self.data.extend(tmp_data)
                logging.info(f'Loaded {len(tmp_data)} samples from {file_path}.')
        logging.info(f'=============Loaded total {len(self.data)} samples from {opt.ppo_pretrain_data_path}.=============')

        self.size = len(self.data)

        if accelerator and self.accelerator.use_distributed:
            self.data = self.data[self.accelerator.process_index::self.accelerator.num_processes]


    def load_data(self, file_path: str):
        with open(file_path, 'r') as f:
            data: List[List[str]] = json.load(f)

        output: List[Tuple[List[str], str]] = []

        for turn in data:
            if not isinstance(turn, list) or len(turn) < 2 or not all(turn):
                continue
            output.append(turn)

        del data
        return output

    def format(self, sample: Tuple[List[str], str]) -> Dict[str, Any]:
        # original text concat special prompt: human prompt and assistant prompt
        import pdb; pdb.set_trace()
        language_predicted = langid.classify(context)[0].strip()  
        context = [get_special_prompt(i, self.opt) + u for i, u in enumerate(sample)]
            
        context_vec = self.tokenizer.encode(
            self.tokenizer.eos_token.join(context) + self.tokenizer.eos_token,
            add_special_tokens=True
        )
        
        text_vec = context_vec[:self.opt.max_seq_len]
        loss_mask = []
        cnt = 0
        for v in text_vec:
            loss_mask.append(cnt % 2)
            cnt += int(v == self.tokenizer.eos_token_id)

        output = {
            'text_vec': text_vec,
            'loss_mask': loss_mask,
        }

        return output

    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = dict()
        batch_text_vec = torch.tensor(pad_sequences(
            [sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.pad_token_id, pad_left=False
            ), dtype=torch.long)
        loss_mask = torch.tensor(pad_sequences(
            [sample['loss_mask'] for sample in batch_samples], pad_value=0, pad_left=False
            ), dtype=torch.bool)
   
        batch.update({
            'text_vec': batch_text_vec,
            'loss_mask': loss_mask
        })
        
        return batch
            
    def batch_generator(self):
        while True:
            for batch in super().batch_generator():
                if len(batch) == self.batch_size:
                    yield batch
            if self.mode != 'train':
                break

