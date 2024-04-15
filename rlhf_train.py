import os
from argparse import ArgumentParser
from src.utils import *
from setting import *
from tokenizer_model import ChatGLMTokenizer
from src.share import *
import random
from src.rlhf.train import ppo_train_reward, ppo_train_policy

def get_model(opt):
    model_path, state_dict, lora_path, lora_state_dict = read_ckpt(opt.model_path)
    model, tokenizer = init_model(opt)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    load_weight(model, state_dict, lora_state_dict, opt.merge_lora_on_load, strict=False)
    model=model.half().eval()
    model.to(opt.device)
    if opt.compile:
        print("Compiling the model...")
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    return model, tokenizer


def ppo_train(opt, accelerator):
    save_dir = opt.model_path.replace('pretrain', 'ppo_train')
    model_name = opt.model_path.split('.')[0]
    log_dir = os.path.join(save_dir, f'{model_name}_log.log')
    # if os.path.exists(log_dir):
    #     os.remove(log_dir) 
    logger = get_logger(log_dir)

    # fix seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # model_dir = os.path.dirname(opt.model_path)
    # opt.config = os.path.join(model_dir, 'config.yaml')
    # if not os.path.exists(opt.config):
    #     opt.config = os.path.join(model_dir, 'config_ds.yaml')

    # opt, config = parser_model_config(opt)
    # model, tokenizer = get_model(opt)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(opt.model_path)
    tokenizer = AutoTokenizer.from_pretrained(opt.model_path)

    # first stage: train reward
    print(f'train reward model')
    ppo_train_reward(opt, model, tokenizer, accelerator)

    # first stage: train policy
    print(f'train policy model')
    ppo_train_policy(opt, model, tokenizer, accelerator)


def dpo_train(opt, accelerator):
    save_dir = opt.model_path.replace('pretrain', 'dpo_train')
    model_name = opt.model_path.split('.')[0]
    log_dir = os.path.join(save_dir, f'{model_name}_log.log')
    # if os.path.exists(log_dir):
    #     os.remove(log_dir) 
    logger = get_logger(log_dir)

    # fix seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print(f'dpo train model')


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--rlhf_type", type=str, default='ppo', choices=['ppo', 'dpo']) 

    opt = get_parser_args(parser)
    # opt.model_path = 'out/pretrain_layer18_seqlen1024_dim1536_accum64_h12_hkv2/epoch_0_step200.pth'
    opt.model_path = './models/opt-125m'
    
    from accelerate import Accelerator
    accelerator = Accelerator(split_batches=True)

    if opt.rlhf_type =='ppo':
        ppo_train(opt, accelerator)
    else:
        dpo_train(opt, accelerator)