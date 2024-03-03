import os
import time
from contextlib import nullcontext
import numpy as np
import torch
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
from src.data.dataset_sft import SFTDataset
from src.data.dataset_pretrain import PretrainDataset
import torch.nn.functional as F
from tokenizer_model import ChatGLMTokenizer
from src.share import *
from src.utils import *
from setting import *


def sft_epoch(epoch,ddp,opt,train_loader,optimizer,model_opt,scaler,ctx,logger):
    iter_per_epoch=len(train_loader)
    start_time=time.time()
    
    ave_time = []
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        single_time_start=time.time()
        
        X=X.to(opt.device)
        Y=Y.to(opt.device)
        loss_mask=loss_mask.to(opt.device)
        
        lr = get_lr(epoch*iter_per_epoch+step, opt) if opt.decay_lr else opt.learning_rate

        if opt.use_deepspeed:
            output = model_opt(X, Y)
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(output.loss*loss_mask)/loss_mask.sum()
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            # backward pass, with gradient scaling if training in fp16
            model_opt.backward(loss)
            model_opt.step()
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # and using the GradScaler if data type is float16
            #for micro_step in range(grad_accum_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model_opt.require_backward_grad_sync = 0 == opt.grad_accum_steps - 1
            
            with ctx:
                output = model_opt(X, Y)
                loss = F.cross_entropy(output.logits.view(-1, output.logits.size(-1)), 
                                       Y.view(-1), 
                                       ignore_index=0,reduce=False)
                loss_mask = loss_mask.view(-1)
                loss = torch.sum(loss*loss_mask)/loss_mask.sum()
                # loss = raw_model.last_loss
                #loss = loss / grad_accum_steps
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        
            if((step+1) % opt.grad_accum_steps)==0:
                # clip the gradient
                if opt.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_opt.parameters(), opt.grad_clip)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
            
        single_time_end=time.time()
        ave_time.append(single_time_end - single_time_start)
        if len(ave_time) > 50:
            del(ave_time[0])
        # print(f'model train ave time: {round(mean(ave_time),6)} s')

        #打印日志
        if step % opt.log_interval == 0:
            spend_time=time.time()-start_time
            logger.info(
                    'Epoch:[{}/{}] ({}/{}) loss:{:.3f} lr:{:.7f}  epoch_time: {} min.'.format(
                        epoch,
                        opt.max_epoch, 
                        step, 
                        iter_per_epoch,
                        loss.item(), 
                        lr,
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))


def ft_model(opt, model_name_path):
    master_process, ddp_local_rank,ctx= init_ddp(ddp, opt)
    
    print(f'**************model_path: {opt.model_path}**************')

    if opt.use_deepspeed:
        print(f"====================use deepspeed====================")
        # 并行环境初始化
        ds_config = read_deepspeed_config(opt)
        if master_process:
            print(ds_config)

    #init model
    model=init_model(opt, train_flag=True)
    model_path, state_dict, lora_path, lora_state_dict = read_ckpt(model_name_path)
    load_weight(model, state_dict)
    model.to(opt.device)
    if opt.ft_type == 'lora':
        from src.loralib.utils import mark_only_lora_as_trainable
        mark_only_lora_as_trainable(model)
    
    if master_process:
        model.print_params()
    
    # optimizer
    optimizer = configure_optimizers(model, opt.weight_decay, opt.learning_rate, 
                                     (opt.beta1, opt.beta2), opt.device, use_fused=opt.ft_type != 'lora')
    
    if opt.use_deepspeed:
        import deepspeed
        # deepspeed初始化
        deepspeed.init_distributed()
        model_opt, optimizer_opt, _, _ = deepspeed.initialize(
            config=ds_config,
            model=model,
            optimizer=optimizer,
            model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
        )
    else:
        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(opt.dtype == 'float16'))
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.max_epoch, 
                                                                        T_mult=1, eta_min=1e-6, last_epoch=-1)
        
        # compile the model
        if opt.compile:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model) # requires PyTorch 2.0

        # wrap model into DDP container
        if ddp:
            # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
            # construction time since NCCL does not support `ComplexFloat`
            prefix = "_orig_mod." if opt.compile else ""
            model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
            model_opt = DDP(model, device_ids=[ddp_local_rank])
            model = model_opt.module
        else:
            model_opt=model
            
    #-----init dataloader------
    tokenizer = ChatGLMTokenizer(vocab_file=opt.vocab_file)

    print(f"====================prepear dataset====================")

    if opt.rope_scaling_factor > 1.0:
        train_ds = SFTDataset(opt.sft_long_data_path_train, tokenizer, max_length=opt.max_seq_len)
        val_ds = SFTDataset(opt.sft_long_data_path_val, tokenizer, max_length=opt.max_seq_len)
    else:
        train_ds = SFTDataset(opt.sft_data_path, tokenizer, max_length=opt.max_seq_len)
        val_ds = PretrainDataset(opt.valid_data_path, max_length=opt.max_seq_len)

    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
        sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=opt.batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )

    print(f"====================sft_epoch====================")

    model_save_type = 'lora' if opt.ft_type =='lora' else 'all'
    
    # sft loop
    best_val_loss = 0.0
    val_loss = 0.0
    for epoch in range(opt.max_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
    
        sft_epoch(epoch,ddp,opt,train_loader,optimizer,model_opt,scaler,ctx,logger)
        val_loss=valid_epoch(model, val_loader, opt, logger, ctx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info('best val_loss: {} best_epoch: {} '.format(best_val_loss,epoch))
            if master_process:  #一般用0，当然，可以选任意的rank保存。
                save_path = '{}/pretrain_{}_ft_best_{}.pth'.format(save_dir,model_name.split('.')[0], epoch)
                save_model(model, save_path, model_save_type, opt.merge_lora_to_save)

        if master_process:  #一般用0，当然，可以选任意的rank保存。
            save_path = '{}/pretrain_{}_ft_epoch_{}.pth'.format(save_dir,model_name.split('.')[0],epoch)
            save_model(model, save_path, model_save_type, opt.merge_lora_to_save)

    if ddp:
        destroy_process_group()

# I/O
if __name__=="__main__":
    opt = get_parser_args()
    if opt.use_deepspeed:
        opt.config = 'config/config_ds.yaml'
    else:
        opt.config = 'config/config.yaml'
    opt, config = parser_all_config(opt)
    opt.ft_type = 'full_ft'
    opt.model_path = './out/pretrain_layer28_seqlen1024_dim1024_accum64_h16_hkv8'  # 输入文件夹

    # 遍历全部sft处理
    if opt.use_deepspeed:
        save_config_file_name = 'config_ds.yaml'
        if opt.ft_type == 'lora':
            save_dir = opt.model_path.replace('pretrain', 'ft_lora_ds')
        else:
            save_dir = opt.model_path.replace('pretrain', 'ft_full_ds')
    else:
        save_config_file_name = 'config.yaml'
        if opt.ft_type == 'lora':
            save_dir = opt.model_path.replace('pretrain', 'ft_lora')
        else:
            save_dir = opt.model_path.replace('pretrain', 'ft_full')

    if os.path.exists(os.path.join(opt.model_path, 'config.yaml')):
        opt.config = os.path.join(opt.model_path, 'config.yaml')
    else:
        opt.config = os.path.join(opt.model_path, 'config_ds.yaml')

    opt, config = parser_model_config(opt)

    if opt.use_deepspeed:
        train_params = dict()
        train_params['use_deepspeed'] = opt.use_deepspeed
        config['train_params'] = train_params
    
    set_fine_tuning_paras_to_config(opt, config)

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # 保存一份参数
    with open(os.path.join(save_dir, save_config_file_name), "w") as file:
        import yaml
        file.write(yaml.dump(config))

    model_list = os.listdir(opt.model_path)
    for model_ in model_list:
        if model_.endswith('.pth'):

            model_name = model_.split('.')[0]
            model_name_path = os.path.join(opt.model_path, model_)

            log_dir = os.path.join(save_dir,f'{model_name}_log.log')
            # if os.path.exists(log_dir):
            #     os.remove(log_dir) 
            logger = get_logger(log_dir)

            ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?

            # -----------------------------------------------------------------------------
            config_keys = [
                k
                for k, v in globals().items()
                if not k.startswith("_") and isinstance(v, (int, float, bool, str))
            ]
            # exec(open("configurator.py").read())  # overrides from command line or config file
            # config = {k: globals()[k] for k in config_keys}  # will be useful for logging
            # -----------------------------------------------------------------------------
            ft_model(opt, model_name_path)
