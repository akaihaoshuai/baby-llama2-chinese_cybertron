import torch
import os
from tqdm import tqdm
import math
from typing import Literal
from argparse import ArgumentParser
from src.model_runner import init_model
from src.utils import read_config
from src.data.dataset_pretrain import PretrainDataset
from src.data.dataset_sft import SFTDataset
from argparse import ArgumentParser

BASE_LR = 3e-4  # 1.5e-4 for 30B-70B models
BASE_BS = 4_000_000  # from llama paper

# https://github.com/hiyouga/LLaMA-Factory/blob/main/scripts/cal_lr.py
def calculate_lr(
    model_name_or_path: str = "/storage/home/pretrainLab/chenhongkai/data/models/xiaotian/sft_Qwen2-7B_xiaotian_sft_20240624-1645",
    train_file: str = "config/train.yaml",
    stage: Literal["pt", "sft"] = "sft",
    # cutoff_len: int = 1024,  # i.e. maximum input length during training
    ddp : bool = False
):
    r"""
    Calculates the optimal learning rate for 7B/13B models using LLaMA's hyper-parameters.
    Usage: python cal_lr.py --model_name_or_path path_to_model --dataset alpaca_en --cutoff_len 1024
    """
    config_file = os.path.join(model_name_or_path, "config.yaml")
    model_config = read_config(config_file)
    _, tokenizer = init_model(model_config, flag='train')
    pretrain_config = read_config(train_file)
    
    if stage == 'pt':
        train_ds = PretrainDataset(pretrain_config['train_data_path'], 
                                max_length=model_config['max_seq_len'],
                                memmap=True)
    else:
        train_ds = SFTDataset(pretrain_config['sft_data_path'], 
                              max_length=model_config['max_seq_len'], 
                              tokenizer=tokenizer)

    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=pretrain_config['batch_size'],
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0 if os.name == 'nt' else 4,
        sampler=train_sampler)

    valid_tokens, total_tokens = 0, 0
    for X, Y, _ in tqdm(train_loader):
        valid_tokens += torch.sum(Y != -100).item()
        total_tokens += torch.numel(Y)

    batch_max_len = model_config['max_seq_len'] * pretrain_config['batch_size']  # max tokens in a batch
    valid_ratio = valid_tokens / total_tokens
    batch_valid_len = batch_max_len * valid_ratio
    lr = BASE_LR * math.sqrt(batch_valid_len / BASE_BS)  # lr ~ sqrt(batch_size)
    print(
        "Optimal learning rate is {:.2e} for valid ratio% {:.2f} and effective batch size {:.2f}".format(
            lr, valid_ratio * 100, batch_valid_len
        )
    )

# I/O
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='./out/pretrain_layer12_dim768_seq768', help="path to config")
    parser.add_argument("--train_file", type=str, default="config/train.yaml", help="path to config")
    parser.add_argument("--stage", type=str, choices=["pt", "sft"], default="sft", help="path to config")
    args = parser.parse_args()

    calculate_lr(args.model_name_or_path, args.train_file, args.stage)