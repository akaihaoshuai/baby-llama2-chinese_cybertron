import os
import time
import numpy as np
import torch
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.dataset_pretrain import PretrainDataset
from numpy import *
from src.utils import *
from src.share import *
from setting import *

#To run with DDP on 4 gpus on 1 node, example:
# torchrun --standalone --nproc_per_node=4 pretrain.py OR python -m torch.distributed.launch --nproc_per_node=4 pretrain.py
def pretrain_epoch(epoch, model_opt, raw_model,
                   train_loader, val_loader, 
                   scaler, optimizer, 
                   opt, ctx, master_process):
    
    start_time=time.time()
    iter_per_epoch=len(train_loader)
    best_val_loss = 1e9
    ave_time = []
    for step, (X, Y) in enumerate(train_loader):
        single_time_start=time.time()

        X=X.to(opt.device)
        Y=Y.to(opt.device)

        lr = get_lr(epoch*iter_per_epoch+step, opt) if opt.decay_lr else opt.learning_rate
        lr = lr*(1.0 + (opt.grad_accum_steps-1)*0.1)
                
        if opt.use_deepspeed:
            output = model_opt(X, Y)
            model_opt.backward(output.loss)
            model_opt.step()
            loss = output.loss
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model_opt.require_backward_grad_sync = 0 == opt.grad_accum_steps - 1
            
            with ctx:
                output = model_opt(X, Y)
                # loss.reduction ='mean':
                loss = output.loss / opt.grad_accum_steps
            
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            loss = loss.item() * opt.grad_accum_steps
        
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

        if step > 0 and step % opt.eval_iters == 0 and val_loader is not None:
            val_loss = valid_model(model_opt, val_loader, logger, opt.device, ctx)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info('best val_loss: {} best_epoch: {} '.format(best_val_loss,epoch))
                
                if master_process:  #一般用0，当然，可以选任意的rank保存。
                    save_model(raw_model, '{}/best.pth'.format(save_dir), opt.merge_lora_to_save)


        if step > 0 and step %opt.save_iters == 0:
            if master_process:  #一般用0，当然，可以选任意的rank保存。
                save_model(raw_model, '{}/epoch_{}_step{}.pth'.format(save_dir,epoch, step), opt.merge_lora_to_save)
            # model_opt.save_checkpoint('{}/epoch_{}.pth'.format(save_dir,epoch), ckpt_id, client_sd = client_sd)
        

        #打印日志
        if step % opt.log_iters == 0:
            spend_time=time.time()-start_time
            logger.info(
                    'Epoch:[{}/{}] ({}/{}) loss:{:.3f} lr:{:.7f}  epoch_time: {} min.'.format(
                        epoch,
                        opt.max_epoch, 
                        step, 
                        iter_per_epoch,
                        loss, 
                        lr,
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))
        

def pretrain_model(opt):
    master_process, ddp_local_rank, ctx = init_ddp(ddp, opt)

    if opt.use_deepspeed:
        print(f"====================use deepspeed====================")
        ds_config = read_deepspeed_config(opt)
        if master_process:
            print(ds_config)
    
    #init model
    model, _ = init_model(opt, train_flag=True)
    if master_process:
        model.print_params()
   
    model.to(opt.device)

    # optimizer
    optimizer = configure_optimizers(model, opt.weight_decay, 
                                     opt.learning_rate, 
                                     (opt.beta1, opt.beta2), 
                                     opt.optimizer_type, 
                                     opt.device)
    
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
        scaler = None
    else:
        # initialize a GradScaler. If enabled=False scaler is a no-op
        # 混合精度训练、在内存中用FP16做储存和乘法从而加速计算，而用FP32做累加避免舍入误差。
        scaler = torch.cuda.amp.GradScaler(enabled=(opt.dtype == 'float16'))
        #
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.max_epoch, T_mult=1, eta_min=1e-6, last_epoch=-1)
        
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
            model_opt= model              
            
    print(f"====================prepear dataset====================")

    #-----init dataloader------
    train_ds = PretrainDataset(opt.train_data_path, max_length=opt.max_seq_len,memmap=True,use_print=master_process)
    val_ds = PretrainDataset(opt.valid_data_path, max_length=opt.max_seq_len,memmap=True,use_print=master_process)
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
        num_workers=0 if os.name == 'nt' else 4,
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

    print(f"====================pretrain_epoch====================")

    warmup_epoch=1
    
    # training loop
    best_val_loss = 1e9
    val_loss = 0.0
    for epoch in range(opt.max_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pretrain_epoch(epoch, model_opt, model,
                       train_loader, val_loader, 
                       scaler, optimizer, 
                       opt, ctx, master_process)
        
        val_loss = valid_model(model_opt, val_loader, logger, opt.device, ctx)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info('best val_loss: {} best_epoch: {} '.format(best_val_loss,epoch))
            
            if master_process:  #一般用0，当然，可以选任意的rank保存。
                save_model(model, '{}/best.pth'.format(save_dir), opt.merge_lora_to_save)

        if master_process:  #一般用0，当然，可以选任意的rank保存。
            save_model(model, '{}/epoch_{}.pth'.format(save_dir,epoch), opt.merge_lora_to_save)
        # model_opt.save_checkpoint('{}/epoch_{}.pth'.format(save_dir,epoch), ckpt_id, client_sd = client_sd)

    if ddp:
        destroy_process_group()
    
# I/O
if __name__=="__main__":
    opt = get_parser_args()
    opt.model_config = 'config/config.yaml'
    opt.train_config = 'config/train.yaml'
    opt, model_config, train_config = parser_all_config(opt)

    # -----------------------------------------------------------------------------
    config_keys = [
        k for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    # exec(open("configurator.py").read())  # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    save_name=f'pretrain_layer{opt.n_layers}_seqlen{opt.max_seq_len}_dim{opt.hidden_size}_accum{opt.grad_accum_steps}_h{opt.n_heads}_hkv{opt.n_kv_heads}'
    save_dir =os.path.join(opt.out_dir , save_name)
    os.makedirs(opt.out_dir, exist_ok=True)
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir, exist_ok=True)

    # 保存一份参数
    with open(os.path.join(save_dir,'config.yaml'), "w") as file:
        import yaml
        file.write(yaml.dump(model_config))

    log_dir = os.path.join(save_dir,'log.log')
    # if os.path.exists(log_dir):
    #     os.remove(log_dir) 

    logger = get_logger(log_dir)
    # various inits, derived attributes, I/O setup
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?

    pretrain_model(opt)