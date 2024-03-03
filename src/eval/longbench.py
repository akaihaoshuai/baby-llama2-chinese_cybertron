import torch
import os
import json
import numpy as np
import random
from tqdm import tqdm
import jsonlines

from src.eval.metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()        
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, opt, model_name):
    preds = []

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        # tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        x = tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.special_tokens['<eos>']]
        x = (torch.tensor(x, dtype=torch.long, device=opt.device)[None, ...])

        if len(x) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(x[:half], skip_special_tokens=True)+tokenizer.decode(x[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        # input = tokenizer(prompt, truncation=False, return_tensors="pt").to(opt.device)
        x = tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.special_tokens['<eos>']]
        x = (torch.tensor(x, dtype=torch.long, device=opt.device)[None, ...])

        context_length = x.shape[-1]
        # print(f'=========length: {context_length}. =========')
        y = model.generate(x, 2, opt.max_new_tokens, temperature=opt.temperature, top_k=opt.top_k)[0]

        pred = tokenizer.decode(y, skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def longbench_eval_func(data_path, opt, model, tokenizer, model_path_name):
    seed_everything(opt.seed) # 1337/42
    
    eval_longbench_e = True
    if eval_longbench_e:  # LongBench-E
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:  # LongBench
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open(os.path.join(data_path,"dataset2prompt.json"), "r"))
    dataset2maxlen = json.load(open(os.path.join(data_path,"dataset2maxlen.json"), "r"))
    
    # predict on each dataset
    dir_name = os.path.splitext(model_path_name)[0]+'_longBench'

    scores = dict()
    for dataset in datasets:
        data = []
        if eval_longbench_e:
            with open(os.path.join(data_path, dataset + '_e.jsonl'), "r+", encoding="utf8") as f:
                for item in jsonlines.Reader(f):
                    data.append(item)

            # data = load_dataset('data', f"{dataset}_e", split='test')
            if not os.path.exists(f"{dir_name}_pred_e"):
                os.makedirs(f"{dir_name}_pred_e")
            out_path = f"{dir_name}_pred_e/{dataset}.jsonl"
        else:
            with open(os.path.join(data_path, dataset + '.jsonl'), "r+", encoding="utf8") as f:
                for item in jsonlines.Reader(f):
                    data.append(item)

            # data = load_dataset('data', dataset, split='test')
            if not os.path.exists(f"{dir_name}_pred"):
                os.makedirs(f"{dir_name}_pred")
            out_path = f"{dir_name}_pred/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        max_length = [3500,7500,15500,31500]
        preds = get_pred(model, tokenizer, data, max_length[0], max_gen, prompt_format, dataset, opt ,model_name='')
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')

        predictions, answers, lengths = [], [], []
        for pred in preds:
            predictions.append(pred["pred"])
            answers.append(pred["answers"])
            all_classes = pred["all_classes"]
            if "length" in pred:
                lengths.append(pred["length"])

        if eval_longbench_e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
        print(f"LongBench {dataset} accuracy: {score}")

    sum_scores={'0-4k': 0, '4-8k': 0, '8k+': 0}
    for key in scores.keys():
        sum_scores['0-4k'] += scores[key]['0-4k'] / len(scores)
        sum_scores['4-8k'] += scores[key]['4-8k'] / len(scores)
        sum_scores['8k+']  += scores[key]['8k+']  / len(scores)

    print("LongBench Average accuracy '0-4k': {:.3f}".format(sum_scores['0-4k']))
    print("LongBench Average accuracy '4-8k': {:.3f}".format(sum_scores['4-8k']))
    print("LongBench Average accuracy '8k+' : {:.3f}".format(sum_scores['8k+']))

    return scores, sum_scores
    
