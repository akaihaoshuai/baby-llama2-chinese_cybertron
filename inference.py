import os
from argparse import ArgumentParser
from src.utils import *
from setting import *
from src.share import *


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


def inference(opt):
    if os.path.isdir(opt.model_path):
        model_dir = opt.model_path
        opt.model_path = [os.path.join(model_dir, file) for file in os.listdir(model_dir) if file.endswith('.pth') or file.endswith('.bin')][0]
    else:
        model_dir = os.path.dirname(opt.model_path)

    opt.model_config = os.path.join(model_dir, 'config.yaml')
    if not os.path.exists(opt.model_config):
        opt.model_config = os.path.join(model_dir, 'config.json')

    opt, _ = parser_model_config(opt)
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