import os
import math
from contextlib import nullcontext
import torch
from src.models.model_loader import _get_model_architecture
from torch.distributed import init_process_group
import logging
import inspect
import numpy as np

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

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type, use_fused=True):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    # param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if (p.dim() >= 2 and p.requires_grad)]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or p.requires_grad==False]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"[optimizers] num decayed parameter tensors: {num_decay_params} parameters")
    print(f"[optimizers] num non-decayed parameter tensors {num_nodecay_params} parameters")
    

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
    

def init_model(opt, train_flag=False):
    # model init
    if opt.init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        model = _get_model_architecture(opt.model_type)(opt, train_flag=train_flag)
    elif opt.init_from == "resume":
        print(f"Resuming training from {opt.model_path}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(opt.model_path, "best.pth")
        checkpoint = torch.load(ckpt_path, map_location=opt.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
            opt[k] = checkpoint_model_args[k]
        # create the model
        model = _get_model_architecture(opt.model_type)(opt, train_flag=train_flag)
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
        master_process = ddp_local_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        #assert grad_accum_steps % ddp_world_size == 0
        #grad_accum_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_local_rank=0

    tokens_per_iter = opt.grad_accum_steps * ddp_world_size * opt.batch_size * opt.max_seq_len
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"breaks down as: {opt.grad_accum_steps} \
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


@torch.no_grad()
def valid_model(model, val_loader, logger, ctx=None):
    losses = []
    model.eval()

    for _, (X, Y) in enumerate(val_loader):
        X=X.to(model.device)
        Y=Y.to(model.device)
        if ctx is not None:
            with ctx:
                output = model(X, Y)
        else:
            output = model(X, Y)
        losses.append(output.loss.item())

    model.train()
    val_loss=np.mean(losses)
    
    logger.info('valid loss = {:.4f}'.format(val_loss))

    return val_loss
