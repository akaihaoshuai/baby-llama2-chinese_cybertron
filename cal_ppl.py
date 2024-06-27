import os
import torch
import json
from tqdm import tqdm
from typing import Any, Dict, Literal, Optional, Sequence
from argparse import ArgumentParser

from src.model_runner import init_model
from src.utils import read_config
from src.data.dataset_pretrain import PretrainDataset
from src.data.dataset_sft import SFTDataset

# https://github.com/hiyouga/LLaMA-Factory/blob/main/scripts/cal_ppl.py
def cal_ppl(
    model_name_or_path: str,
    train_file: str = "config/train.yaml",
    # stage: Literal["pt", "sft", "rm"] = "sft",
    dataset: str = "data/sft_data.csv",
    ddp : bool = False,
    device: str = 'cuda',
):
    r"""
    Calculates the ppl on the dataset of the pre-trained models.
    Usage: python cal_ppl.py --model_name_or_path path_to_model --save_name ppl.json
    """
    config_file = os.path.join(model_name_or_path, "config.yaml")
    model_config = read_config(config_file)
    model, tokenizer = init_model(model_config, flag='train')
    model.to(device)
    train_config = read_config(train_file)

    # if stage == 'pt':
    #     train_ds = PretrainDataset(train_config['train_data_path'], 
    #                                max_length=model_config['max_seq_len'],
    #                                memmap=True)
    # else:
    train_ds = SFTDataset(dataset, 
                          max_length=model_config['max_seq_len'], 
                          tokenizer=tokenizer)

    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_config['batch_size'],
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0 if os.name == 'nt' else 4,
        sampler=train_sampler)

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    total_ppl = 0
    perplexities = []
    batch: Dict[str, "torch.Tensor"]
    with torch.no_grad():
        for X, Y, loss_mask in tqdm(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            outputs = model(X,Y)
            shift_logits: "torch.Tensor" = outputs["logits"][..., :-1, :]
            shift_labels: "torch.Tensor" = Y[..., 1:]
            loss_mask = shift_labels != -100
            flatten_logits = shift_logits.contiguous().view(shift_labels.size(0) * shift_labels.size(1), -1)
            flatten_labels = shift_labels.contiguous().view(-1)
            token_logps: "torch.Tensor" = criterion(flatten_logits, flatten_labels)
            token_logps = token_logps.contiguous().view(shift_logits.size(0), -1)
            sentence_logps = (token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            total_ppl += sentence_logps.exp().sum().item()
            perplexities.extend(sentence_logps.exp().tolist())

    save_name = os.path.join(model_name_or_path, "ppl.json")
    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(perplexities, f, indent=2)

    print("Average perplexity is {:.2f}".format(total_ppl / len(perplexities)))
    print("Perplexities have been saved at {}.".format(save_name))

# I/O
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='./out/pretrain_layer12_dim768_seq768', help="path to config")
    parser.add_argument("--train_file", type=str, default="config/train.yaml", help="path to config")
    parser.add_argument("--dataset", type=str, default="data/sft_data.csv", help="path to config")
    args = parser.parse_args()

    cal_ppl(args.model_name_or_path, args.train_file, args.dataset)