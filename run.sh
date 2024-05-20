# python data_prepare.py
# CUDA_VISIBLE_DEVICES=1 python ./data_prepare.py

# 重新训练tokenizer
# CUDA_VISIBLE_DEVICES=1 python ./train_tokenizer.py

use_accelerate=true
use_nohup=false

if [ use_accelerate == true ] ; then
    echo "[LLM] use accelerate"
    if [ use_nohup == true ] ; then
        echo "[LLM] use nohup"
        CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python -m torch.distributed.launch --nproc_per_node=8 --use_env pretrain.py >out/pretrain_1_log
        CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python -m torch.distributed.launch --nproc_per_node=8 --use_env sft.py >out/sft_log
        CUDA_VISIBLE_DEVICES=0 nohup python benchmark.py >out/benchmark_log
    else
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env pretrain.py
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env sft.py
        CUDA_VISIBLE_DEVICES=1 python benchmark.py
    fi
else  # deepspeed
    echo "[LLM] use deepspeed"
    if [ use_nohup == true ] ; then
        echo "[LLM] use nohup"
        CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus=4 pretrain.py  --use_deepspeed True >out/pretrain_ds_log
        CUDA_VISIBLE_DEVICES=1,2,3,4 nohup deepspeed --num_gpus=4 sft.py  --use_deepspeed True >out/sft_ds_log
        CUDA_VISIBLE_DEVICES=10 nohup python benchmark.py >out/benchmark_ds_log
    else
        CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 pretrain.py --use_deepspeed True
        CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 sft.py --use_deepspeed True
        CUDA_VISIBLE_DEVICES=1 python benchmark.py
    fi
fi


# 量化模型
# CUDA_VISIBLE_DEVICES=0 python quant_model.py --model ./out/pretrain.ckpt --dataset wikitext2 --wbits 4

# 推理
CUDA_VISIBLE_DEVICES=0 python web_inference.py --model_path ./out/pretrain.ckpt
CUDA_VISIBLE_DEVICES=0 python inference.py --model_path ./out/pretrain.ckpt

# RLHF
# CUDA_VISIBLE_DEVICES=0 rlhf_train.py