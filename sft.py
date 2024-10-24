import os
import time
import torch
import numpy as np
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from argparse import ArgumentParser
import torch.nn.functional as F
import yaml
from src.utils import *
from src.data.dataset_sft import SFTDataset
from src.model_runner import init_model, eval_model, set_model_eval, set_model_train

#------------------------------------------------------------------------------
def train_epoch(epoch, sft_config, master_process, lisa_ft=None):
    start_time=time.time()
    for step, (X, Y,loss_mask) in enumerate(train_loader):
        set_model_train(model, lisa_ft, step)

        X=X.to(device)
        Y=Y.to(device)
        loss_mask=loss_mask.to(device)
        if sft_config['sft_params']['decay_lr'] :
            lr = get_lr(epoch*iter_per_epoch+step, sft_config['sft_params']) 
        else :
            lr = sft_config['sft_params']['lr']

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # and using the GradScaler if data type is float16
        #for micro_step in range(sft_config['grad_accum_steps']):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = 0 == sft_config['grad_accum_steps'] - 1
        with ctx:
            logits = model(X, Y).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1, reduce=False)
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss*loss_mask)/loss_mask.sum()
            #loss = raw_model.last_loss
            #loss = loss / sft_config['grad_accum_steps']
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        
        # clip the gradient
        if sft_config['sft_params']['grad_clip'] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), sft_config['sft_params']['grad_clip'])
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        #打印日志
        if step % sft_config['log_interval'] == 0 and master_process:
            set_model_eval(model)
            eval_model(raw_model, ctx)

            spend_time=time.time()-start_time
            logger.info(
                    'Epoch:[{}/{}]({}/{}) loss: {:.3f}. lr: {:.7f}. epoch_remain_time: {}min.'.format(
                        epoch,
                        sft_config['max_epoch'], 
                        step, 
                        iter_per_epoch,
                        loss.item(), 
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))
            
#------------------
@torch.no_grad()
def valid_epoch(epoch, val_loader):
    global best_val_loss
    losses = []
    set_model_eval(model)
    for _, (X, Y) in enumerate(val_loader):
        X=X.to(device)
        Y=Y.to(device)
        with ctx:
            logits, loss = model(X, Y)
        losses.append(loss.item())
    set_model_train(model)
    val_loss=np.mean(losses)
    #
    logger.info('valid loss = {:.4f}'.format(val_loss))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        logger.info('best val_loss: {} best_epoch: {} '.format(best_val_loss,epoch))
        torch.save(raw_model.state_dict(),'{}/best.pth'.format(save_dir))
    #
    return val_loss


# I/O
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./out/pretrain_layer12_dim512_seq512', help="path to config")
    parser.add_argument("--sft_file", type=str, default='./config/train.yaml', help="path to config")
    args = parser.parse_args()
    
    model_path_dir = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path)
    config_file = os.path.join(model_path_dir, "config.yaml")
    model_config = read_config(config_file)
    sft_config = read_config(args.sft_file)

    save_dir =os.path.join(sft_config['out_dir'], 
                           f'sft_layer{model_config["n_layers"]}_dim{model_config["hidden_dim"]}_seq{model_config["max_seq_len"]}')
    
    os.makedirs(save_dir, exist_ok=True)

    # 保存一份参数
    with open(os.path.join(save_dir,'config.yaml'), "w") as file:
        file.write(yaml.dump(model_config))
    
    lora_config = None
    if sft_config['sft_params']['type'] == 'lora' or sft_config['sft_params']['type'] == 'dora':
        lora_config = read_config(args.sft_file.replace('train.yaml', 'lora.yaml'))
        with open(os.path.join(save_dir,'lora.yaml'), "w") as file:
            file.write(yaml.dump(lora_config))

    logger = get_logger(os.path.join(save_dir,'log.log'))
    # various inits, derived attributes, I/O setup
   # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    master_process, ddp_world_size, ddp_local_rank, device = init_ddp(ddp, sft_config['device'])

    tokens_per_iter = sft_config['grad_accum_steps'] * ddp_world_size * sft_config['batch_size'] * model_config["max_seq_len"]
    if master_process:
        print_rank_0(f"tokens per iteration will be: {tokens_per_iter:,}")
        print_rank_0(f"breaks down as: {sft_config['grad_accum_steps']} grad accum steps * {ddp_world_size} processes * {sft_config['batch_size']} batch size * {model_config['max_seq_len']} max seq len")

    if master_process:
        os.makedirs(sft_config['out_dir'], exist_ok=True)
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    # ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[sft_config['dtype']]
    ctx = get_ctx(device_type)
    
    best_val_loss = 1e9

    #init model
    model, tokenizer = init_model(model_config, model_path_dir, 
                                  lora_config=lora_config, 
                                  flag=sft_config['sft_params']['type'])
    model.to(device)
    set_model_train(model)
    print_rank_0('***************model****************')
    print_rank_0(model)

    lisa_ft = None
    if sft_config['sft_params']['type'].upper() == 'LISA':
        from src.ft_opt.lisa import LISA_ft
        assert sft_config['sft_params']['interval_steps'] % sft_config['grad_accum_steps'] == 0
        lisa_ft = LISA_ft(sft_config['sft_params']['act_layers'], sft_config['sft_params']['interval_steps'], model)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(sft_config['dtype'] == 'float16'))
    # optimizer
    optimizer = model.configure_optimizers(sft_config['sft_params']['weight_decay'], 
                                           sft_config['sft_params']['lr'], 
                                           (sft_config['sft_params']['beta1'], sft_config['sft_params']['beta2']), 
                                           device_type)
    
    #-----init dataloader------
    train_ds = SFTDataset(sft_config['sft_data_path'], max_length=model_config['max_seq_len'], tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=sft_config['batch_size'],
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )
    # val_ds = PretrainDataset(data_path_list, max_length=model_config['max_seq_len'])
    # val_loader = torch.utils.data.DataLoader(
    #     val_ds,
    #     batch_size=sft_config['batch_size'],
    #     pin_memory=False,
    #     drop_last=False,
    #     shuffle=False,        
    #     num_workers=0,
    # )

    iter_per_epoch=len(train_loader)
    # compile the model
    if sft_config['compile']:
        print_rank_0("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0  #sudo apt-get install build-essential
    # wrap model into DDP container
    if ddp:
        # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support `ComplexFloat`
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model # unwrap DDP container if needed

    # training loop
    for epoch in range(sft_config['max_epoch']):
        train_epoch(epoch, sft_config, master_process, lisa_ft)
        if master_process:
            torch.save(raw_model.state_dict(),'{}/epoch_{}.pth'.format(save_dir,epoch))
    if ddp:
        destroy_process_group()
