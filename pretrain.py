import os
import time
import torch
from argparse import ArgumentParser
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.dataset_pretrain import PretrainDataset
from src.utils import *
from src.model_runner import init_model, eval_model, set_model_eval, set_model_train

def train_epoch(epoch, pretrain_config, master_process):
    start_time=time.time()
    for step, (X, Y) in enumerate(train_loader):
        X=X.to(device)
        Y=Y.to(device)
        if pretrain_config['train_params']['decay_lr'] :
            lr = get_lr(epoch*iter_per_epoch+step, pretrain_config['train_params']) 
        else :
            lr = pretrain_config['train_params']['lr']

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # and using the GradScaler if data type is float16
        #for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = 0 == pretrain_config['grad_accum_steps'] - 1
        with ctx:
            loss = model(X, Y).last_loss
            loss = loss / pretrain_config['grad_accum_steps']
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        
        if (step + 1) % pretrain_config['grad_accum_steps'] == 0:
            # clip the gradient
            if pretrain_config['train_params']['grad_clip'] != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), pretrain_config['train_params']['grad_clip'])
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

        #打印日志
        if step % pretrain_config['log_interval'] == 0 and master_process:
            set_model_eval(model)
            eval_model(raw_model, ctx)
            set_model_train(model)

            spend_time=time.time()-start_time
            logger.info(
                    'Epoch:[{}/{}]({}/{}) loss: {:.3f}. lr: {:.7f}. epoch_remain_time: {}min.'.format(
                        epoch,
                        pretrain_config['max_epoch'], 
                        step, 
                        iter_per_epoch,
                        loss.item(), 
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))
        #
        if step > 0 and step % pretrain_config['save_interval'] == 0 and master_process:
            set_model_eval(model)
            torch.save(raw_model.state_dict(),'{}/iter_{}.pth'.format(save_dir,int(step+epoch*iter_per_epoch)))
            set_model_train(model)

# @torch.no_grad()
# def valid_epoch(epoch):
#     global best_val_loss
#     losses = []
#     set_model_eval(model)
#     for _, (X, Y) in enumerate(val_loader):
#         X=X.to(device)
#         Y=Y.to(device)
#         with ctx:
#             logits, loss = model(X, Y)
#         losses.append(loss.item())
#     set_model_train(model)
#     val_loss=np.mean(losses)
#     #
#     logger.info('valid loss = {:.4f}'.format(val_loss))
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         logger.info('best val_loss: {} best_epoch: {} '.format(best_val_loss,epoch))
#         torch.save(raw_model.state_dict(),'{}/best.pth'.format(save_dir))
#     #
#     return val_loss


# I/O
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, default='./config/config.yaml', help="path to config")
    parser.add_argument("--pretrain_file", type=str, default='./config/train.yaml', help="path to config")
    args = parser.parse_args()

    model_config = read_config(args.config_file)
    pretrain_config = read_config(args.pretrain_file)

    save_dir =os.path.join(pretrain_config['out_dir'], 
                           f'pretrain_layer{model_config["n_layers"]}_dim{model_config["hidden_dim"]}_seq{model_config["max_seq_len"]}')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存一份参数
    with open(os.path.join(save_dir,'config.yaml'), "w") as file:
        import yaml
        file.write(yaml.dump(model_config))

    logger = get_logger(os.path.join(save_dir,'log.log'))
    # various inits, derived attributes, I/O setup
   # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    
    master_process, ddp_world_size, ddp_local_rank, device = init_ddp(ddp, pretrain_config['device'])

    tokens_per_iter = pretrain_config['grad_accum_steps'] * ddp_world_size * pretrain_config['batch_size'] * model_config['max_seq_len']
    if master_process:
        print_rank_0(f"tokens per iteration will be: {tokens_per_iter:,}")
        print_rank_0(f"breaks down as: {pretrain_config['grad_accum_steps']} grad accum steps * {ddp_world_size} processes * {pretrain_config['batch_size']} batch size * {model_config['max_seq_len']} max seq len")
        os.makedirs(pretrain_config['out_dir'], exist_ok=True)

    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    # ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[pretrain_config['dtype']]
    ctx = get_ctx(device_type)
    
    best_val_loss = 1e9
    
    #init model
    model, _ = init_model(model_config, flag='train')
    model.to(device)
    set_model_train(model)
    print_rank_0('***************model****************')
    print_rank_0(model)

    
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(pretrain_config['dtype'] == 'float16'))
    # optimizer
    optimizer = model.configure_optimizers(pretrain_config['train_params']['weight_decay'], 
                                           pretrain_config['train_params']['lr'], 
                                           (pretrain_config['train_params']['beta1'], 
                                            pretrain_config['train_params']['beta2']), 
                                            device_type)
    
    # compile the model
    if pretrain_config['compile']:
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

     #-----init dataloader------
    train_ds = PretrainDataset(pretrain_config['train_data_path'], 
                               max_length=model_config['max_seq_len'],
                               memmap=True)
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
        sampler=train_sampler
    )
    # val_ds = PretrainDataset(data_path_list, max_length=256)
    # val_loader = torch.utils.data.DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     pin_memory=False,
    #     drop_last=False,
    #     shuffle=False,        
    #     num_workers=0,
    # )

    # training loop
    iter_per_epoch=len(train_loader)
    for epoch in range(pretrain_config['max_epoch']):
        train_epoch(epoch, pretrain_config, master_process)
        #val_loss=valid_epoch(epoch)
        if master_process:  #一般用0，当然，可以选任意的rank保存。
            torch.save(raw_model.state_dict(),'{}/epoch_{}.pth'.format(save_dir,epoch))
    if ddp:
        destroy_process_group()
