import logging
import os
import yaml
import json
import torch
import math
from contextlib import nullcontext
from torch.distributed import init_process_group

from src.models.model_args import ModelArgs
from src.models.cybertron import Cybertron
from src.chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer


def init_model(model_config=None, model_path=None, tokenizer=None):
    # model init
    if model_path is None:
        # init a new model from scratch
        print("Initializing a new model from scratch")
        model_args = ModelArgs(**model_config) if model_config is not None else ModelArgs()
        if tokenizer is None:
             tokenizer = ChatGLMTokenizer()
        
        model_args.bos_id = tokenizer.tokenizer.bos_id
        model_args.eos_id = tokenizer.tokenizer.eos_id
        model_args.pad_id = tokenizer.tokenizer.pad_id
        model = Cybertron(model_args)
    else:
        # resume training from a checkpoint.
        model_path_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        epoch_bin_path_list = [sub_path for sub_path in os.listdir(model_path_dir) if 'epoch' in sub_path]
        
        max_epoch = -1
        for bin_path in epoch_bin_path_list:
            epoch_num = int(bin_path.split('_')[-1].split('.')[0])
            if max_epoch < epoch_num:
                 max_epoch = epoch_num
        
        ckpt_path = os.path.join(model_path, f'epoch_{max_epoch}.pth')
        print(f'load model from path: {ckpt_path}.')

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'model_config' in checkpoint:
            checkpoint_model_config = checkpoint["model_config"]
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
                model_config[k] = checkpoint_model_config[k]

        # create the model
        model_args = ModelArgs(**model_config) if model_config is not None else ModelArgs()
        if tokenizer is None:
             tokenizer = ChatGLMTokenizer()
        
        model_args.bos_id = tokenizer.tokenizer.bos_id
        model_args.eos_id = tokenizer.tokenizer.eos_id
        model_args.pad_id = tokenizer.tokenizer.pad_id
        model = Cybertron(model_args)

        if 'model' in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

    return model

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



def eval_model(model, ctx=None):
    from src.chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
    tokenizer=ChatGLMTokenizer()

    model.eval()
    device = next(model.parameters()).device

    data = [
        {"question": "最近我在办公室坐久了会感到头晕，请问这是什么原因?有什么缓解办法吗？", "target": ""},
        # {"question": "前列腺囊肿的症状是什么？", "target": ""},
        # {"question": "请问，世界上最大的动物是什么？", "target": ""},
    ]
    if ctx is None:
         ctx = get_ctx(device)

    for p in data:
        # run generation
        prompt=p['question']
        x=tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.special_tokens['<bos>']]
        x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
        target = p['target']
        with torch.no_grad():
            with ctx:
                y = model.generate(x)
                answer=tokenizer.decode(y[0].tolist())
                answer=answer.replace(prompt,'')
                print('[prompt]:',prompt)
                print('[answer]:',answer)
                print('---------------')


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