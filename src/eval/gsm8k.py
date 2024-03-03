import os
import json
import torch
from tqdm import tqdm
import numpy as np
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

class GSM8K:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        opt,
        model_path_name,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.opt = opt
        self.model_path_name = model_path_name

    def run(self, data_path, shot: int):
        results, accs = {}, {}

        dir_name = os.path.splitext(self.model_path_name)[0]+'_Ceval'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # run all task
        for task_name in self.TASK2DESC:
            print("================================================================")
            print(f"run task: {task_name}")
            result, acc = self.run_single_task(data_path, task_name, shot)
            results[task_name] = result
            accs[task_name] = acc

            result_path = os.path.join(dir_name, f"{task_name}.json")
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"save result to {result_path}")

        average_acc = sum(accs.values()) / len(accs)
        accs['average'] = average_acc

        # results
        acc_path = os.path.join(dir_name, "ceval_acc.json")
        with open(acc_path, "w") as f:
            json.dump(accs, f, indent=2)
        print(f"Ceval average acc: {average_acc}\n")

        return accs, average_acc

    def run_single_task(self, data_path, task_name: str, shot: int):
        import os
        dataset = dict()
        if os.path.exists(data_path):
            import pandas as pd
            for name in ('val', 'dev'):
                csv_data = pd.read_csv(
                    os.path.join(os.path.join(data_path, name), task_name + '_' + name + '.csv'))
                questions = []
                for idx in range(len(csv_data)):
                    data_que = dict()
                    data_que["id"] = csv_data.id[idx]
                    data_que["question"] = csv_data.question[idx]
                    data_que["A"] = csv_data.A[idx]
                    data_que["B"] = csv_data.B[idx]
                    data_que["C"] = csv_data.C[idx]
                    data_que["D"] = csv_data.D[idx]
                    data_que["answer"] = csv_data.answer[idx]
                    questions.append(data_que)

                dataset[name] = questions
        else:
            from datasets import load_dataset
            dataset = load_dataset(self.DATA_PATH, task_name)

        # tmp_ = dataset[split]
        results = []
        acc = 0

        for data in tqdm(dataset['val']):
            prompt = f"以下是中国关于{self.TASK2DESC[task_name]}考试的单项选择题，请选出其中的正确答案。\n"
            if shot != 0:
                if isinstance(dataset["dev"], list):
                    import random
                    random.shuffle(dataset["dev"])
                    shuffled = dataset["dev"]
                else:
                    shuffled = dataset["dev"].shuffle()
                for i in range(min(shot, len(shuffled))):
                    prompt += "\n" + self.build_example(shuffled[i], with_answer=True)
            prompt += "\n" + self.build_example(data, with_answer=False)

            # input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
            # logits = self.model(
            #         input_ids=input_ids,
            #     ).logits[:,-1].flatten()

            x=self.tokenizer.encode(prompt,add_special_tokens=False)+[self.tokenizer.special_tokens['<eos>']]
            x = (torch.tensor(x, dtype=torch.long, device=self.opt.device)[None, ...])
            logits = self.model(x).logits[0][0]

            candidate_logits = [logits[self.tokenizer(label,add_special_tokens=False).input_ids[-1]] for label in ["A", "B", "C", "D"]]
            candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
            probs = (
                torch.nn.functional.softmax(
                    candidate_logits,
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(probs)]

            results.append(
                {
                    "prompt": prompt,
                    "correct": answer == data["answer"].strip().upper(),
                    "answer": answer,
                }
            )
            acc += answer == data["answer"].strip().upper()
        acc /= len(dataset['val'])
        return results, acc

    def build_example(self, data, with_answer: bool = True):
        question = data["question"]
        choice = "\n".join(
            [
                "A. " + data["A"],
                "B. " + data["B"],
                "C. " + data["C"],
                "D. " + data["D"],
            ]
        )
        answer = data["answer"].strip().upper() if with_answer else ""
        return f"{question}\n{choice}\n答案: {answer}"
    