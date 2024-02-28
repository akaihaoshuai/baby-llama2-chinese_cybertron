import os
import time
import numpy as np
import torch
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.dataset_pretrain import PretrainDataset
from numpy import *
import deepspeed
from src.utils import *
from src.share import *
from setting import *

#To run with DDP on 4 gpus on 1 node, example:
# torchrun --standalone --nproc_per_node=4 pretrain.py OR python -m torch.distributed.launch --nproc_per_node=4 pretrain.py
def pretrain_epoch_ds(epoch, model_engine, train_loader, optimizer, opt):
    start_time=time.time()
    iter_per_epoch=len(train_loader)

    ave_time = []
    for step, (X, Y) in enumerate(train_loader):
        single_time_start=time.time()

        X=X.to(opt.device)
        Y=Y.to(opt.device)
        
        _, loss, _ = model_engine(X, Y)
        
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # backward pass, with gradient scaling if training in fp16
        model_engine.backward(loss)
        model_engine.step()

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
                        loss, 
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))
        


def pretrain_model(opt):
    master_process,ddp_local_rank,ctx=init_ddp(ddp, opt)

    # 并行环境初始化
    ds_config = read_deepspeed_config(opt)
    if master_process:
        print(ds_config)
    
    os.makedirs(opt.out_dir, exist_ok=True)
    #init model
    model=init_model(opt)
    optimizer = configure_optimizers(model, opt.weight_decay, opt.learning_rate, 
                                     (opt.beta1, opt.beta2), opt.device, use_fused=False)

    if master_process:
        model.print_params()


    # deepspeed初始化
    deepspeed.init_distributed()
    model_engine, optimizer_engine, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        optimizer=optimizer,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
    )
        
    print(f"====================prepear dataset====================")

    #-----init dataloader------
    train_ds = PretrainDataset(opt.train_data_path,max_length=opt.max_seq_len,memmap=True,use_print=opt.local_rank==0)
    val_ds = PretrainDataset(opt.valid_data_path,max_length=opt.max_seq_len,use_print=opt.local_rank==0)
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

    # training loop
    best_val_loss = 1e9
    for epoch in range(opt.max_epoch):
        pretrain_epoch_ds(epoch, model_engine, train_loader, optimizer, opt)
        val_loss=valid_epoch(model, val_loader, opt, logger)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info('best val_loss: {} best_epoch: {} '.format(best_val_loss,epoch))

            client_sd=dict()
            client_sd['step'] = epoch
            ckpt_id = val_loss
            save_model(model, '{}/best.pth'.format(save_dir))
            # model_engine.save_checkpoint('{}/best.pth'.format(save_dir), ckpt_id, client_sd = client_sd)

        save_model(model, '{}/epoch_{}.pth'.format(save_dir,epoch))
        # model_engine.save_checkpoint('{}/epoch_{}.pth'.format(save_dir,epoch), ckpt_id, client_sd = client_sd)


# I/O
if __name__=="__main__":
    opt = get_parser_args()
    opt.config = 'config/config_ds.yaml'
    opt,config = parser_model_config(opt)
    opt.lora_path = None   # 预训练肯定不加载lora

    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    # exec(open("configurator.py").read())  # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    save_name=f'pretrain_ds_layer{opt.n_layers}_seqlen{opt.max_seq_len}_dim{opt.dim}_bs{opt.batch_size}_accum{opt.grad_accum_steps}_h{opt.n_heads}_hkv{opt.n_kv_heads}'
    save_dir =os.path.join(opt.out_dir , save_name)
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir, exist_ok=True)

    # 保存一份参数
    with open(os.path.join(save_dir,'config.yaml'), "w") as file:
        import yaml
        file.write(yaml.dump(config))

    log_dir = os.path.join(save_dir,'log.log')
    # if os.path.exists(log_dir):
    #     os.remove(log_dir) 
        
    logger = get_logger(log_dir)
    # various inits, derived attributes, I/O setup
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?

    pretrain_model(opt)