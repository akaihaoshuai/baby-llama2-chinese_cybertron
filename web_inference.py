import os
import sys
import math
import torch
import argparse
from transformers import TextIteratorStreamer
from threading import Thread
import gradio as gr
from src.utils import *
from setting import *


title = "baby-llama2-chinese for Long-context LLMs"

description = """
<font size=4>
This is the online demo of baby-llama2-chinese. \n
github: https://github.com/akaihaoshuai/baby-llama2-chinese_fix \n
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
    model, tokenizer, temperature=0.6, top_p=0.9, max_new_tokens=4096, use_cache=True
):
    def response(material, question):
        if material is None:
            return "Only support txt file."

        if not material.name.split(".")[-1]=='txt':
            return "Only support txt file."

        material = read_txt_file(material.name)
        prompt_no_input = PROMPT_DICT["prompt_no_input_llama2"]
        prompt = prompt_no_input.format_map({"instruction": material + "\n%s" % question})

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        if len(inputs['input_ids'][0]) > 32768:
            return "This demo supports tokens less than 32768, while the current is %d. Please use material with less tokens."%len(inputs['input_ids'][0])
        torch.cuda.empty_cache()
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(**inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            streamer=streamer,
            )

        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text
        return generated_text

    return response

def main(opt):
    # if args.flash_attn:
    #     replace_llama_attn(inference=True)
    from inference import get_model
    model_dir = os.path.dirname(opt.model_path)
    opt.config = os.path.join(model_dir, 'config.yaml')
    if not os.path.exists(opt.config):
        opt.config = os.path.join(model_dir, 'config_ds.yaml')

    opt, config = parser_model_config(opt)

    orig_ctx_len = getattr(config, "max_seq_len", None)
    if orig_ctx_len and opt.max_new_tokens > orig_ctx_len:
        opt.rope_scaling_factor = float(math.ceil(opt.max_new_tokens / orig_ctx_len))
        opt.rope_scaling_type = "linear"

    model, tokenizer = get_model(opt)
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        
    model=model.half().eval()
    respond = build_generator(model, tokenizer, temperature=opt.temperature, top_p=opt.top_p,
                              max_new_tokens=opt.max_seq_len, use_cache=True)

    demo = gr.Interface(
        respond,
        inputs=[
            gr.File(type="file", label="Input material txt"),
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
    demo.launch(server_name=opt.host, server_port=opt.port, show_error=True, share=True)

if __name__ == "__main__":
    opt = get_parser_args()
    opt.model_path = 'out/fft_layer28_seqlen1024_dim1024_bs2_accum64_h16_hkv8/pretrain_epoch_1_ft_epoch_0.pth'
    
    main(opt)
