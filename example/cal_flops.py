import torch
import os
from argparse import ArgumentParser
from deepspeed.accelerator import get_accelerator  # type: ignore
from deepspeed.profiling.flops_profiler import get_model_profile  # type: ignore

from src.model_runner import init_model
from src.utils import read_config

# https://github.com/hiyouga/LLaMA-Factory/blob/main/scripts/cal_flops.py
def calculate_flops(
    model_name_or_path: str,
    batch_size: int = 1,
    seq_length: int = 256,
):
    r"""
    Calculates the flops of pre-trained models.
    Usage: python cal_flops.py --model_name_or_path path_to_model --batch_size 1 --seq_length 512
    """
    with get_accelerator().device(0):
        config_file = os.path.join(model_name_or_path, "config.yaml")
        model_config = read_config(config_file)
        model, _ = init_model(model_config, flag='train')

        fake_input = torch.ones((batch_size, seq_length), dtype=torch.long, device=model.device)
        input_dict = {"input_ids": fake_input, "labels": fake_input.clone()}
        flops, macs, params = get_model_profile(model, kwargs=input_dict, print_profile=True, detailed=True)
        print("FLOPs:", flops)
        print("MACs:", macs)
        print("Params:", params)

# I/O
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='./out/pretrain_layer12_dim768_seq768', help="path to config")
    parser.add_argument("--batch_size", type=int, default=1, help="path to config")
    parser.add_argument("--seq_length", type=int, default=256, help="path to config")
    args = parser.parse_args()

    calculate_flops(args.model_name_or_path, args.batch_size, args.seq_length)