import os
from argparse import ArgumentParser
from src.utils import *
from setting import *
from src.models.Jerry import Jerry
from tokenizer_model import ChatGLMTokenizer
from src.share import *


def get_model(opt):
    model_path, state_dict, lora_path, lora_state_dict = read_ckpt(opt.model_path)
    model = init_model(opt)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    load_weight(model, state_dict, lora_state_dict, opt.merge_lora_on_load, strict=False)

    model.eval()
    model.to(opt.device)
    if opt.compile:
        print("Compiling the model...")
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

        # load the tokenizer
    tokenizer=ChatGLMTokenizer(vocab_file=opt.vocab_file)

    return model, tokenizer


def inference(opt):
    model_dir = os.path.dirname(opt.model_path)
    opt.config = os.path.join(model_dir, 'config.yaml')
    if not os.path.exists(opt.config):
        opt.config = os.path.join(model_dir, 'config_ds.yaml')

    opt, config = parser_model_config(opt)

    model, tokenizer = get_model(opt)
    
    x=tokenizer.encode(opt.prompt, add_special_tokens=False)
    x = (torch.tensor(x, dtype=torch.long, device=opt.device)[None, ...])
    y = model.generate(x, 
                       max_new_tokens=opt.max_new_tokens, 
                       temperature=opt.temperature, 
                       top_k=opt.top_k)
    predict=tokenizer.decode(y)

    print(f'prompt: {opt.prompt}. /n response: {predict}')


if __name__=="__main__":
    opt = get_parser_args()
    opt.model_path = 'out/pretrain_layer18_seqlen1024_dim1536_accum64_h12_hkv2/epoch_0_step200.pth'
    
    inference(opt)