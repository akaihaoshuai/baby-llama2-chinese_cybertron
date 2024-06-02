import os
import sys
import math
import torch
from transformers import TextIteratorStreamer
from threading import Thread
from argparse import ArgumentParser
import gradio as gr
from src.utils import *
from src.model_runner import init_model

title = "baby-llama2-chinese for Long-context LLMs"

description = """
<font size=4>
This is the online demo of baby-llama2-chinese. \n
github: https://github.com/akaihaoshuai/baby-llama2-chinese_cybertron \n
If multiple users are using it at the same time, they will enter a queue, which may delay some time. \n
**Inputs**: <br>
- **Input material txt** and **Question** are required. <br>
**Note**: <br>
- There are 10 book-related examples and 5 paper-related examples, 15 in total.<br>
- Note that only txt file is currently support.\n
**Example questions**: <br>
&ensp; Please summarize the book in one paragraph. <br>
&ensp; Please tell me that what high-level idea the author want to indicate in this book. <br>
&ensp; Please describe the relationship among the roles in the book. <br>
&ensp; Please summarize the paper in one paragraph. <br>
&ensp; What is the main contribution of this paper? <br>
Hope you can enjoy our work!
</font>
"""

PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2":(
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
}


def read_txt_file(material_txt):
    content = ""
    with open(material_txt) as f:
        for line in f.readlines():
            content += line
    return content

def build_generator(
    model, tokenizer, use_cache=True, use_srteam=False
):
    def response(material, question):
        material = ''
        if not (material is None or material == ''):
            if not material.name.split(".")[-1]=='txt':
                return "Only support txt file."
            material = read_txt_file(material.name)

        prompt_no_input = PROMPT_DICT["prompt_no_input_llama2"]
        prompt = prompt_no_input.format_map({"instruction": material + "\n%s" % question})
        device = next(model.parameters()).device

        # if use_srteam:
        #     # x=tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
            
        #     # if len(x['input_ids'][0]) > 32768:
        #     #     return "This demo supports tokens less than 32768, while the current is %d. Please use material with less tokens."%len(inputs['input_ids'][0])
        #     # torch.cuda.empty_cache()
            
        #     # streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        #     # generate_kwargs = dict(**x,
        #     #     use_cache=use_cache,
        #     #     streamer=streamer,
        #     #     )

        #     # t = Thread(target=model.generate, kwargs=generate_kwargs)
        #     # t.start()
            
        #     # generated_text = ""
        #     # for new_text in streamer:
        #     #     generated_text += new_text
        #     #     yield generated_text
        # else:
        x = tokenizer.encode(prompt, add_special_tokens=False) + [tokenizer.special_tokens['<eos>']]
        x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
        outputs = model.generate(x)
        generated_text = tokenizer.decode(outputs[0])
        generated_text = generated_text.replace(prompt, '')

        return generated_text

    return response

def main(args):
    # if args.flash_attn:
    #     replace_llama_attn(inference=True)
    
    model_path_dir = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path)
    config_file = os.path.join(model_path_dir, 'config.yaml')
    model_config = read_config(config_file)

    model, tokenizer = init_model(model_config)
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        
    model=model.half().eval().cuda()
    respond = build_generator(model, tokenizer, use_cache=True)

    demo = gr.Interface(
        respond,
        inputs=[
            gr.File(type="filepath", label="Input material txt"),
            gr.Textbox(lines=1, placeholder=None, label="Question"),
        ],
        outputs=[
            gr.Textbox(lines=1, placeholder=None, label="Text Output"),
        ],
        title=title,
        description=description,
        allow_flagging="auto",
    )

    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, show_error=True, share=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./out/pretrain_layer10_dim512_seq256', help="path to config")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8898)
    args = parser.parse_args()

    main(args)
