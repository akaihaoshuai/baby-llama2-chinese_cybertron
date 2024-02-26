# python data_prepare.py

use_accelerate=false
use_nohup=false

if [ use_accelerate == true ] ; then
    echo "[LLM] use accelerate"
    if [ use_nohup == true ] ; then
        echo "[LLM] use nohup"
        CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python -m torch.distributed.launch --nproc_per_node=8 --use_env pretrain.py >out/pretrain_1_log
        CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python -m torch.distributed.launch --nproc_per_node=8 --use_env fine_tuning.py >out/fine_tuning_log
        CUDA_VISIBLE_DEVICES=0 nohup python eval.py >out/eval_log
    else
        CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --use_env pretrain.py
        CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --use_env fine_tuning.py
        CUDA_VISIBLE_DEVICES=1 python eval.py
    fi
else  # deepspeed
    echo "[LLM] use deepspeed"
    if [ use_nohup == true ] ; then
        echo "[LLM] use nohup"
        CUDA_VISIBLE_DEVICES=1,2,3,4 nohup deepspeed --num_gpus=4 pretrain_ds.py  >out/pretrain_ds_log
        CUDA_VISIBLE_DEVICES=1,2,3,4 nohup deepspeed --num_gpus=4 fine_tuning_ds.py >out/fine_tuning_ds_log
        CUDA_VISIBLE_DEVICES=10 nohup python eval.py >out/eval_ds_log
    else
        CUDA_VISIBLE_DEVICES=1,2,3,4 deepspeed --num_gpus=4 pretrain_ds.py
        CUDA_VISIBLE_DEVICES=1,2,3,4 deepspeed --num_gpus=4 fine_tuning_ds.py
        CUDA_VISIBLE_DEVICES=1 python eval.py
    fi
fi
