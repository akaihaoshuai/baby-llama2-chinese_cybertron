import argparse
import json
import os
import time
from tqdm import tqdm
import torch
import argparse
import json
import jsonlines
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir)

DATA_PATH = "data/2wikimqa_e.jsonl"

def load_datasets(prompt_token_length, tokenizer, system_prompt_num):
    prompts = []
    with jsonlines.open(DATA_PATH) as reader:
        current_tokens = []

        for line in reader:
            # Tokenize current line
            tokens = tokenizer(line['context']+"\n"+line['input'], add_special_tokens=False)['input_ids']

            if len(tokens) >= prompt_token_length:
                # 如果当前行的 tokens 足够长，直接裁剪并加入 prompts
                prompts.append(tokens[:prompt_token_length])
            else:
                # 否则，尝试拼接
                current_tokens.extend(tokens)

                # 如果拼接后的长度足够，加入 prompts
                if len(current_tokens) >= prompt_token_length:
                    prompts.append(current_tokens[:prompt_token_length])
                    current_tokens = current_tokens[prompt_token_length:]

            # 如果已经得到足够的 prompts，停止处理
            if len(prompts) >= system_prompt_num:
                break

        # 如果还有未处理完的 tokens，且 prompts 数量不足 n，补充最后一个 prompt
        if current_tokens and len(prompts) < system_prompt_num:
            prompts.append(current_tokens[:prompt_token_length])

    # Decode tokens back to text
    prompts_text = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts]
    return prompts_text


def hf_inference(model, tokenizer, data_list, answer_length):
    # Format the input as a chat template
    sample = [data for data in data_list]
    # Tokenize input and generate output
    input_ids = tokenizer(sample, return_tensors='pt', add_special_tokens=False).to("cuda")
    start_time = time.time()
    outputs = model.generate(**input_ids, 
                             max_new_tokens=answer_length, 
                             return_dict_in_generate=False, 
                             output_scores=False, 
                             use_cache=True, 
                             num_beams=1, 
                             do_sample=False)
    end_time = time.time()
    
    response_data_list = []
    for prompt, input_id, output in zip(sample, input_ids['input_ids'], outputs):
        response_data = dict()
        response_data['prompt'] = prompt
        response_data['input_token_len'] = input_id.shape[0]
        response_data['text'] = tokenizer.decode(output[-(output.shape[0]-input_id.shape[0]):])
        response_data['output_token_len'] = output.shape[0]-input_id.shape[0]
        response_data['cost_time'] = end_time - start_time
        response_data_list.append(response_data)

    return response_data_list



def hf_benchmark(model, tokenizer, prompt_length, answer_length, system_prompt_num, save_dir, batch_size: int = 1, tp: int = 1):
    dataset = load_datasets(prompt_length, tokenizer, system_prompt_num)

    t1 = time.time()
    results = []
    # 每次从dataset中获取
    for _idx in tqdm(range(0, len(dataset), batch_size), desc=f"bs={batch_size}, len_in={prompt_length}, len_out={answer_length}"):
        results.extend(hf_inference(model, tokenizer, dataset[_idx:_idx+batch_size], answer_length))
    t2 = time.time()

    save_path = os.path.join(save_dir, f"tp_{tp}_bs_{batch_size}_prompt_{prompt_length}_systemn_{system_prompt_num}_answer_{answer_length}_auto.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({"ave_time": sum([data['cost_time'] for data in results])/len(results), 'predict': results}, fp=f, indent=4, ensure_ascii=False)
    print(f'cost time: {(t2 - t1):.2f}s')
    time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str, default='0')
    parser.add_argument("--save-dir", type=str, default='speed_benchmark')
    parser.add_argument("--model-lists", type=str, nargs="+", default=["/root/autodl-fs/models/Hymba-1.5B-Instruct"])
    args = parser.parse_args()

    tensor_parallel_size = len(args.gpu_id.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    for model_path in args.model_lists:
        os.makedirs(args.save_dir, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", 
                                                     trust_remote_code=True,
                                                     torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        save_dir = os.path.join(args.save_dir, model_path.split('/')[-1])
        os.makedirs(save_dir, exist_ok=True)
        for batch_size in [1]:  # 4, 16
            system_prompt_num = max(batch_size*4, 16)
            for prompt_length in [1024, 4096, 8192, 16384, 32768, 32768, 65536]:    # 2510, 4096, 8192, 16384
                for answer_length in [1]:  # 500, 1000
                    print(f'[model_path] {model_path}.')
                    print(f'[predict] tp:{tensor_parallel_size}, batch_size:{batch_size}, prompt_length:{prompt_length}, answer_length:{answer_length}.')
                    hf_benchmark(model=model,
                                tokenizer=tokenizer,
                                system_prompt_num=system_prompt_num,
                                prompt_length=prompt_length,
                                answer_length=answer_length,
                                batch_size=batch_size,
                                tp=tensor_parallel_size,
                                save_dir=save_dir)