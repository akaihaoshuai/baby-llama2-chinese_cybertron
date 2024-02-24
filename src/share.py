import os
import math
from contextlib import nullcontext
import torch
from src.models.utils import ModelArgs
from src.models.model_loader import _get_model_architecture
from torch.distributed import init_process_group
import logging
from typing import Dict, Sequence
import transformers
import copy
import inspect

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

def tensorboard_logger(loss,epoch):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir='./tensorboard_logs', comment='train_loss')
    # writer.add_image("cat",cat_img_224)
    writer.add_scalars('data/data_group', {'loss': loss}, epoch)
    writer.close()


# -----------------------------------------------------------------------------
def get_lr(it, opt):
    # 1) linear warmup for warmup_iters steps
    if it < opt.warmup_iters:
        return opt.learning_rate * it / opt.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > opt.lr_decay_iters:
        return opt.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - opt.warmup_iters) / (opt.lr_decay_iters - opt.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return opt.min_lr + coeff * (opt.learning_rate - opt.min_lr)

# -----------------------------------------------------------------------------
def get_model_args(opt):
    model_args = dict(
        dim=opt.dim,
        n_layers=opt.n_layers,
        n_heads=opt.n_heads,
        n_kv_heads=opt.n_kv_heads if opt.n_kv_heads > 0 else opt.n_heads,
        vocab_size=opt.vocab_size,#64793,
        multiple_of=opt.multiple_of,
        max_seq_len=opt.max_seq_len,
        use_bias=opt.use_bias,
        dropout=opt.dropout,
        flash_attention = False,
        model_type = 'Model',
    )  # start with model_args from command line

    return model_args

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type, use_fused=True):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    # param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    if use_fused:
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"\nusing fused AdamW: {use_fused} \n")
    else:
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    return optimizer
    

def init_model(opt):
    # model init
    model_args=get_model_args(opt)

    if opt.init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = _get_model_architecture(gptconf.model_type)(gptconf)
    elif opt.init_from == "resume":
        print(f"Resuming training from {opt.model_path}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(opt.model_path, "best.pth")
        checkpoint = torch.load(ckpt_path, map_location=opt.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = ModelArgs(**model_args)
        model = _get_model_architecture(gptconf.model_type)(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    return model

def init_ddp(ddp, opt):
    print(f"====================prepear backend====================")
    if ddp:
        print(f"====================open DistributedDataParallel====================")
        # Check if the operating system is Windows
        if os.name == 'nt':
            # Diff between backends: https://pytorch.org/docs/stable/distributed.html
            init_process_group(backend="gloo")
        else:
            # If the operating system is Linux based, os.name == 'posix'
            init_process_group(backend=opt.backend)
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
        ddp_local_rank=0

    tokens_per_iter = opt.gradient_accumulation_steps * ddp_world_size * opt.batch_size * opt.max_seq_len
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"breaks down as: {opt.gradient_accumulation_steps} \
              grad accum steps * {ddp_world_size} processes * \
              {opt.batch_size} batch size * {opt.max_seq_len} max seq len")

    print(f"====================prepear context====================")
    
    torch.manual_seed(opt.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[opt.dtype]
    ctx = (
        nullcontext()
        if opt.device == "cpu"
        else torch.cuda.amp.autocast()
    )

    return master_process, ddp_local_rank, ctx


# def get_ds_model(model_config, args, activation_checkpointing_config=None):
#     import deepspeed
#     class GPT2ModelPipe(PipelineModule):
#         def __init__(self, model_config, **kwargs):
#             if activation_checkpointing_config:
#                 deepspeed.checkpointing.configure(
#                     None,
#                     partition_activations=activation_checkpointing_config.get("partition_activations", False),
#                     contiguous_checkpointing=activation_checkpointing_config.get("contiguous_memory_optimization", False),
#                     checkpoint_in_cpu=activation_checkpointing_config.get("cpu_checkpointing", False),
#                     num_checkpoints=activation_checkpointing_config.get("number_checkpoints", None),
#                     synchronize=activation_checkpointing_config.get("synchronize_checkpoint_boundary", False),
#                     profile=activation_checkpointing_config.get("profile", False),
#                 )
#             super().__init__(
#                 layers=[
#                     LayerSpec(EmbeddingPipe, model_config.vocab_size + 1, model_config.hidden_size),
#                     *[LayerSpec(ParallelTransformerLayerPipe, model_config, activation_checkpointing_config is not None)
#                         for _ in range(model_config.num_hidden_layers)],
#                     LayerSpec(LayerNormPipe, model_config.hidden_size, model_config.rms_norm_eps),
#                     LayerSpec(LMLayerPipe, model_config.hidden_size, model_config.vocab_size + 1, bias=False),
#                 ],
#                 **kwargs
#             )

#     pp = args.pipe_parallel_size
#     mp = args.model_parallel_size
#     assert args.world_size % (pp * mp) == 0
#     dp = args.world_size // (pp * mp)

#     from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
#     topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=mp, num_dp=dp)
#     # Offset base seeds for the interior pipeline stages.
#     stage_id = topo.get_coord(rank=torch.distributed.get_rank()).pipe
#     if 0 < stage_id < topo.get_dim('pipe') - 1:
#         args.seed = args.seed + (stage_id * mp)

#     return GPT2ModelPipe(model_config,
#                          loss_fn=loss_fn,
#                          topology=topo,
#                          base_seed=args.seed,)

IGNORE_INDEX = -100
PROMPT_FIELD = 'prompt'
OUTPUT_FIELD = 'output'
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    # TODO: batch encode
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            #padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    mode: str
) -> Dict:
    """Preprocess the data by tokenizing."""
    samples = [s + t for s, t in zip(sources, targets)]
    samples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (samples, sources)]
    input_ids = samples_tokenized["input_ids"]
    # FIXME: sentencepiece case
    if mode == "sft":
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
    elif mode == "pretrain":
        labels = copy.deepcopy(input_ids)
    else:
        raise ValueError('Unvalid training mode.')

    # shift
    return dict(
        input_ids=[ids[: -1] for ids in input_ids],
        labels=[lbs[1: ]for lbs in labels]
    )

class DataCollatorForDataset(object):
    """Collate for supervised fine-tuning."""

    mode: str

    def get_attn_mask(self, input_ids):
        """
        Get triangular attention mask for a given sequence length / device.
        """
        bs = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        # lower triangular attention mask
        mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
            bs, 1, seq_length, seq_length
        )
        # convert to binary
        return mask < 0.5

    def get_position_ids(self, input_ids):
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [sample[PROMPT_FIELD] for sample in samples]
        targets = [sample[OUTPUT_FIELD] for sample in samples]

        data_dict = preprocess(sources, targets, self.tokenizer, self.mode)
        input_ids = data_dict["input_ids"]
        labels = data_dict["labels"]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        labels = torch.where(labels == self.tokenizer.pad_token_id, IGNORE_INDEX, labels)

        return (
            (
                input_ids,
                self.get_position_ids(input_ids),
                self.get_attn_mask(input_ids),
            ),
            labels
        )