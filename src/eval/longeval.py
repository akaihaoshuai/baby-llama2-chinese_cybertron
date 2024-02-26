import os
from tqdm import tqdm
import json
import torch
import time
import re
from fastchat.model import load_model, get_conversation_template


# from  https://github.com/DachengLi1/LongChat

def load_testcases(test_file):
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases

def retrieve_from_openai(prompt, model_name, num_retries=10):
    import openai
    import tiktoken
    openai.api_key = os.environ["OPENAI_API_KEY"]
    token_size = len(tiktoken.encoding_for_model(model_name).encode(prompt))
    
    num_retries = 10
    completion = None
    for attempt in range(num_retries):
        backoff = 2 ** (attempt)

        try:    
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt}"}    
                ],
                temperature = 0
            )
            break
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            if attempt == num_retries - 1:
                raise
        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            if attempt == num_retries - 1:
                raise
        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            if attempt == num_retries - 1:
                raise
        except openai.error.Timeout as e:
            print(f"OpenAI API request timed out: {e}")
            if attempt == num_retries - 1:
                raise
        except openai.error.InvalidRequestError as e:
            print(f"Invalid request to OpenAI API: {e}")
            if attempt == num_retries - 1:
                raise
        except openai.error.AuthenticationError as e:
            print(f"Authentication error with OpenAI API: {e}")
            if attempt == num_retries - 1:
                raise
        except openai.error.ServiceUnavailableError as e:
            print(f"OpenAI API service unavailable: {e}")
            if attempt == num_retries - 1:
                raise
        time.sleep(backoff)

    if completion is None:
        print(f"Failed to get response after {num_retries} retries")
        return token_size, -1, "Rate limit"

    response_line = completion.choices[0].message["content"]

    return token_size, response_line

def retrieve_from_anthropic(prompt, model_name, num_retries=10):
    import anthropic
    from anthropic import HUMAN_PROMPT, AI_PROMPT
    client = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])

    completion = client.completion(
        model = model_name,
        max_retries=num_retries,
        max_tokens_to_sample=300,
        temperature=0,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"
    )

    return -1, completion["completion"]

def test_topics_one_sample(model, tokenizer, test_case, output_file, idx, opt):
    prompt = test_case["prompt"]
    topics = test_case["topics"]
    
    x = tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.special_tokens['<eos>']]
    x = (torch.tensor(x, dtype=torch.long, device=opt.device)[None, ...])
    
    y = model.generate(x, 2, opt.max_new_tokens, temperature=opt.temperature, top_k=opt.top_k)
    predict=tokenizer.decode(y[0].tolist())
    predict=predict.replace(prompt,'')
    
    summary = f"Label: {topics[0]}, Predict: {predict}, prompt length: {len(prompt)}".replace('\n', ' ')
    print(summary)

    if idx ==0:
        with open(output_file, "w") as f:
            f.write(summary)
            f.write("\n")
    else:
        with open(output_file, "a+") as f:
            f.write(summary)
            f.write("\n")
    
    return None, len(prompt), summary

def test_lines_one_sample(model, tokenizer, test_case, output_file, idx, opt):
    prompt = test_case["prompt"]
    correct_line = test_case["correct_line"]
    expected_number = test_case["expected_number"]

    x = tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.special_tokens['<eos>']]
    x = (torch.tensor(x, dtype=torch.long, device=opt.device)[None, ...])

    y = model.generate(x, 2, opt.max_new_tokens, temperature=opt.temperature, top_k=opt.top_k)
    predict=tokenizer.decode(y[0].tolist())
    predict=predict.replace(prompt,'')

    # Matching the last digit of the model output
    response_number = re.findall("\d+", predict)
    if response_number is not None and len(response_number) > 0:
        response_number = int(response_number[-1])
    else:
        print(f"Got unparsable result")
        response_number = -1

    summary = f"Label: {expected_number}, Predict: {predict}, Parsed: {response_number}, prompt length: {len(prompt)}".replace('\n', ' ')
    # print(summary)
    if idx ==0:
        with open(output_file, "w") as f:
            f.write(summary)
            f.write("\n")
    else:
        with open(output_file, "a+") as f:
            f.write(summary)
            f.write("\n")
    
    return expected_number == response_number, len(prompt), summary

def longeval_eval_func(data_path, opt, model, tokenizer, model_path_name):
    # if opt.task == "topics":
    scores = dict()

    dir_name = os.path.splitext(model_path_name)[0]+'_LongEval'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # for num_topics in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
    if False:
        for num_topics in [5, 10, 15, 20, 25, 30]:
            print(f"************ Start testing {num_topics} topics per prompt ***********")
            avg_length = 0

            test_file = os.path.join(data_path, f"topics/testcases/{num_topics}_topics.jsonl")
            output_file = os.path.join(dir_name, f"{num_topics}_response.txt")
            
            test_cases = load_testcases(test_file)
            for idx, test_case in tqdm(enumerate(test_cases)):
                _, prompt_length, summary = test_topics_one_sample(model=model, tokenizer=tokenizer, 
                                                                test_case=test_case, output_file=output_file, 
                                                                idx=idx, opt=opt)
                avg_length += prompt_length / len(test_cases)

            scores[f'topics{num_topics}'] = 0
            print(f"************ Finish testing {num_topics} topics per prompt with average prompt length {avg_length} ************")


    # elif opt.task == "lines":
    # for num_lines in [200, 300, 400, 500, 600, 680, 700, 800, 900, 1000, 1100, 1200, 1350]:
    for num_lines in [200, 300, 400, 500, 600]:
        print(f"************ Start testing {num_lines} lines per LRT prompt ************")
        test_file = os.path.join(data_path, f"lines/testcases/{num_lines}_lines.jsonl")
        
        output_file = os.path.join(dir_name, f"{num_lines}_response.txt")
        num_correct = 0
        avg_length = 0

        test_cases = load_testcases(test_file)
        for idx, test_case in tqdm(enumerate(test_cases)):
            correct, prompt_length, summary = test_lines_one_sample(model=model, tokenizer=tokenizer, 
                                                                    test_case=test_case, output_file=output_file, 
                                                                    idx=idx, opt=opt)
            avg_length += prompt_length / len(test_cases)
            num_correct += correct

        accuracy = num_correct / len(test_cases)

        with open(output_file, "a+") as f:
            f.write(f"Accuracy: {accuracy}")

        scores[f'lines_{num_lines}'] = accuracy
        print(f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")

    weighted_acc = sum(scores.values())/len(scores)
    print("LongBench Average accuracy: {:.3f}".format(weighted_acc))

    return scores, weighted_acc
