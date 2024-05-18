import os
import sys
import math
import torch
from argparse import ArgumentParser
from src.utils import *


def main(args):
    model_path_dir = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path)
    config_file = os.path.join(model_path_dir, 'config.yaml')
    model_config = read_config(config_file)

    model=init_model(model_config)
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        
    model=model.half().eval().cuda()
    tokenizer = ChatGLMTokenizer()
    device = next(model.parameters()).device

    x = tokenizer.encode(args.prompt, add_special_tokens=False) + [tokenizer.special_tokens['<eos>']]
    x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
    outputs = model.generate(x)
    generated_text = tokenizer.decode(outputs[0])
    generated_text = generated_text.replace(args.prompt, '')
    print(f'prompt: {args.prompt}. \nanswer: {generated_text}')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./out/pretrain_layer10_dim512_seq256', help="path to config")
    parser.add_argument("--prompt", type=str, default='你好', help="path to config")
    args = parser.parse_args()

    main(args)
