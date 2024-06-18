import os
import sys
import math
import torch
from argparse import ArgumentParser
from src.utils import *
from src.model_runner import init_model


def main(args):
    model_path_dir = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path)
    config_file = os.path.join(model_path_dir, 'config.yaml')
    model_config = read_config(config_file)

    model, tokenizer = init_model(model_config, model_path_dir)
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        
    model=model.half().eval().cuda()
    device = next(model.parameters()).device

    x = tokenizer.encode(args.prompt, add_special_tokens=False) + [tokenizer.special_tokens['<eos>']]
    x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
    if args.return_qk_head_hetmaps:
        outputs, qk_heatmaps = model.generate(x, return_qk_head_hetmaps=args.return_qk_head_hetmaps)
    else:
        outputs = model.generate(x)

    generated_text = tokenizer.decode(outputs)
    generated_text = generated_text.replace(args.prompt, '')
    print_rank_0(f'prompt: {args.prompt}. \nanswer: {generated_text}')

    if args.return_qk_head_hetmaps:
        from src.profile.visualize import display_qk_heatmap_per_head
        text_list = [tokenizer.decode(token) for token in outputs]
        display_qk_heatmap_per_head(qk_heatmaps, text_list, model_path_dir.split('/')[-1])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./out/pretrain_layer8_dim512_seq512', help="path to config")
    parser.add_argument("--prompt", type=str, default='where are you from?', help="path to config")
    parser.add_argument("--return_qk_head_hetmaps", type=bool, default=False, help="save qkhead heatmap")
    args = parser.parse_args()

    main(args)
