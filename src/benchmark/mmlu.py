import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from src.utils import get_ctx, print_rank_0
from src.benchmark.categories import subcategories, categories

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

MMLU_DATA_PATH = 'cais/mmlu'

def format_example(df, idx, choices,include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, choices,k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i,choices)
    return prompt


@torch.no_grad()
def eval_single_subject(shot, subject, model, tokenizer, dev_df, test_df, choices):
    cors = []
    all_probs = []
    device = next(model.parameters()).device
    answers = choices[: test_df.shape[1] - 2]
    ctx = get_ctx(device)

    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = shot
        prompt_end = format_example(test_df, i, choices,include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, choices,k)
        prompt = train_prompt + prompt_end
        # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        x = tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.special_tokens['<eos>']]
        x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
        
        while x.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            x = tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.special_tokens['<eos>']]
            x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])

        label = test_df.iloc[i, test_df.shape[1] - 1]

        # logits = model(
        #     input_ids=input_ids,
        # ).logits[:,-1].flatten()
        with ctx:
            logits = model(x).logits[0][0]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A",add_special_tokens=False).input_ids[-1]],
                        logits[tokenizer("B",add_special_tokens=False).input_ids[-1]],
                        logits[tokenizer("C",add_special_tokens=False).input_ids[-1]],
                        logits[tokenizer("D",add_special_tokens=False).input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print_rank_0("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def mmlu_eval_func(data_path, shot, model, tokenizer, model_path_name, benchmark_dir):
    mmlu_res_name = os.path.join(benchmark_dir, 'MMLU')
    os.makedirs(mmlu_res_name, exist_ok=True)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    for subject in subcategories:
        subject_data_path = os.path.join(data_path, subject)
        if os.path.exists(subject_data_path):
            dev_df = pd.read_csv(
                os.path.join(data_path, "dev", subject + "_dev.csv"), header=None
            )[: shot]
            test_df = pd.read_csv(
                os.path.join(data_path, "test", subject + "_test.csv"), header=None
            )
        else:
            print_rank_0(f'file no existed. load_dataset from huggingface: {MMLU_DATA_PATH}/{subject}')
            dev_df = load_dataset(MMLU_DATA_PATH, subject, split='dev')
            test_df = load_dataset(MMLU_DATA_PATH, subject, split='test')

        choices = ["A", "B", "C", "D"]
        cors, acc, probs = eval_single_subject(shot, subject, model, tokenizer, dev_df, test_df, choices)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(model_path_name)] = cors

        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(model_path_name, choice)] = probs[:, j]
        
        test_df.to_csv(
            os.path.join(mmlu_res_name, "results_{}.csv".format(subject)),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print_rank_0("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print_rank_0("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print_rank_0("Average accuracy: {:.3f}".format(weighted_acc))

    return cat_cors, weighted_acc