import logging
import os
import yaml
import json
import torch
import math
from contextlib import nullcontext
from torch.distributed import init_process_group


def print_rank_0(message: str) -> None:
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
        



def init_ddp(ddp, device):
    if ddp:
        # Check if the operating system is Windows
        if os.name == 'nt':
            # Diff between backends: https://pytorch.org/docs/stable/distributed.html
            init_process_group(backend="gloo")
        else:
            # If the operating system is Linux based, os.name == 'posix'
            init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        #assert gradient_accumulation_steps % ddp_world_size == 0
        #gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_local_rank = 0
        device = device

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    return master_process, ddp_world_size, ddp_local_rank, device

def get_ctx(device_type):
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.cuda.amp.autocast()
    )
    return ctx
         

# -----------------------------------------------------------------------------
def get_lr(it, params):
    # 1) linear warmup for warmup_iters steps
    if it < params['warmup_iters']:
        return params['lr'] * it / params['warmup_iters']
    # 2) if it > lr_decay_iters, return min learning rate
    if it > params['lr_decay_iters']:
        return params['min_lr']
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - params['warmup_iters']) / (params['lr_decay_iters'] - params['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return params['min_lr'] + coeff * (params['lr'] - params['min_lr'])



def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def check_is_processed(data_path):
    if os.path.exists(data_path):
        data_path_list = os.listdir(data_path)
        for data_name in data_path_list:
            if data_name.endswith(".bin"):
                return True

    return False


class Config:
	def __init__(self, entries: dict={}):
		for k, v in entries.items():
			if isinstance(v, dict):
				self.__dict__[k] = Config(v)
			else:
				self.__dict__[k] = v

def read_config(config_path):
    if config_path.endswith('.yaml'):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
    elif config_path.endswith('.json'):
        with open(config_path) as f:
            config = json.load(f)

    # return Config(config)
    return config