## baby-llama2-chinese-fix（wip）
从头预训练+SFT一个小参数量的LLM.

本项目是便于自己学习LLM相关知识所建，实现了一些功能，但没有对全部功能进行详细测试，有些功能可能会有一些bug，比如除GPTQ外的其他量化功能。

目前训练效果上存在一些问题，暂时没有时间排查，可以在原项目上训练。



## 更新记录
>2024.05.18：预训练效果正常，loss<0.3，有一定涌现能力。

>2024.05.16：由于有些累积bug没解决，之前更新全部保存到bug_pause_20240516分支中。重新切换主分支继续更新相关内容，会做更加详细的测试。
如有用到之前更新的内容，也会重新引入功能进行测试。

>2024.03.20：增加RLHF代码，可以运行，效果未验证。[https://zhuanlan.zhihu.com/p/686217468]

>2024.03.20：支持GPTQ量化，可以运行。增加llm.int8/awq/onebit量化代码，但代码未测试，[https://zhuanlan.zhihu.com/p/686161543]

>2024.03.10：增加YaRN/CLEX等位置编码，解决kv_cache的bug。[https://zhuanlan.zhihu.com/p/684907262]

>2024.03.02：支持LoRA训练，根据LongLoRA优化代码，支持SS-Attn

>2024.02.29：支持长度外推，from LLaMA。 [https://zhuanlan.zhihu.com/p/683731440]

>2024.02.24：支持deepspeed训练。[https://zhuanlan.zhihu.com/p/683768690]

>2023.11.02：增加训练tokenizer代码，扩展数据。[https://zhuanlan.zhihu.com/p/664046612]

>2023.10.21：测试falsh attention

>2023.10.13：fork代码，训练实战。[https://zhuanlan.zhihu.com/p/660759033]

<br />



## 训练数据
- Wiki中文百科（25w词条）[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- BaiduBaiKe（563w词条）
[百度网盘](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb)
 提取码: bwvb
- [Medical Dataset](https://huggingface.co/datasets/shibing624/medical/tree/main)

除此之外，为了让模型具备在某一个专有领域的能力，这里选用了“医疗问答”作为切入点，尝试收集了很多的医疗数据和上面的通用语料一起喂给模型。

tips：训练数据在训练的过程中有所扩展，详情可参考知乎文章。

## 中文分词器

采用ChatGLM2的分词器。

## 预训练语料预处理
```python
#脚本里面每一个函数对应一个语料库的预处理，搭建新加语料可以自行扩展。
python data_process.py
#运行结束后，会在./data目录下产生.bin文件
```
数据预处理采取GPT的通用做法，对语料进行提前分词，对一个样本做完分词后在末尾加上一个结束符号，与下一个样本区分开。然后将所有的训练语料拼接成一个数组（np.uint16）以.bin二进制格式存储到磁盘上。如果语料过大，避免内存溢出，可以选择mmap格式。

## 训练自己的分词器
如果要重新训练自己的分词器，可以在data_process.py代码中设置save_all_text为True，会将文本信息汇总保存到本地txt，然后可以训练自己的分词器。
```python
#获取data文件夹下的全部tokenizer_xxx.txt文件。
python train_tokenizer.py
#运行结束后，会在当前目录下产生tokenizer.model文件
```

代码来自：https://github.com/yanqiangmiffy/how-to-train-tokenizer

## SFT样本构建
中文SFT语料最近陆陆续续开源了很多（[bell](https://huggingface.co/datasets/BelleGroup/train_1M_CN)、[MOSS](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data)、[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)等），但是坦白讲，质量都不高，大家可自行下载并需要进行清洗，清洗SFT数据是个耗时耗力的工作。
中文SFT语料网上最近很多，大家自行下载。参考dataset_sft.py即可！

基本逻辑如下：
- prompt和answer之间一定要有一个开始符隔开，然后answer后需要一个结束符。
- 计算loss的时候，对prompt部分的loss进行mask，只计算answer部分的loss即可。

<br />

## 预训练+SFT脚本
参考run.sh

```python
# python data_prepare.py
# CUDA_VISIBLE_DEVICES=1 python data_prepare.py

# 重新训练tokenizer
# CUDA_VISIBLE_DEVICES=1 python train_tokenizer.py

use_accelerate=true
use_nohup=false

if [ use_accelerate == true ] ; then
    echo "[LLM] use accelerate"
    if [ use_nohup == true ] ; then
        echo "[LLM] use nohup"
        CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=8 --use_env pretrain.py >out/pretrain_1_log
        CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=8 --use_env fine_tuning.py >out/fine_tuning_log
        CUDA_VISIBLE_DEVICES=0 nohup python eval.py >out/eval_log
    else
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env pretrain.py
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env fine_tuning.py
        CUDA_VISIBLE_DEVICES=0 python eval.py
    fi
else  # deepspeed  可以训练，但loss暴涨，原因未知，精力问题，暂未排查，建议使用accelerator训练
    echo "[LLM] use deepspeed"
    if [ use_nohup == true ] ; then
        echo "[LLM] use nohup"
        CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus=4 pretrain.py  --use_deepspeed True >out/pretrain_ds_log
        CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus=4 fine_tuning.py  --use_deepspeed True >out/fine_tuning_ds_log
        CUDA_VISIBLE_DEVICES=0 nohup python eval.py >out/eval_ds_log
    else
        CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 pretrain.py --use_deepspeed True
        CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 fine_tuning.py --use_deepspeed True
        CUDA_VISIBLE_DEVICES=0 python eval.py
    fi
fi

# 量化模型
# CUDA_VISIBLE_DEVICES=0 python quant_model.py --model ./out/pretrain.ckpt --dataset wikitext2 --wbits 4

```


根据自己算力的情况合理的调节以下参数，控制模型的计算量和参数量，这是第一版使用的参数
- max_seq_len = 256
- hidden_size = 512
- n_layers = 8
- n_heads = 8

推理脚本可以参考eval.py。
