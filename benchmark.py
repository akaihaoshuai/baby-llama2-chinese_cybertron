"""
Sample from the trained model with PyTorch
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import jieba
from argparse import ArgumentParser
from src.utils import *
from src.chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from src.utils import *
from src.model_runner import init_model


CEVAL_DATA_PATH = 'data/ceval-exam'
MMLU_DATA_PATH = 'data/mmlu'
LONGBENCH_DATA_PATH = 'data/longBench'
LONGEVAL_DATA_PATH = 'data/longEval'
PERPLEXITY_DATA_PATH = 'data/perplexity'
GSM8K_DATA_PATH = 'data/gsm8k'

def compute_bleu(labels, preds, weights=None):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    weights = weights or (0.25, 0.25, 0.25, 0.25)
    return np.mean([sentence_bleu(references=[label],
                                  hypothesis=pred,
                                  smoothing_function=SmoothingFunction().method1,
                                  weights=weights) for label, pred in zip(labels, preds)])


def eval_medical(model, model_path_dir, tokenizer, benchmark_config, logger):
    answer_list=[]
    predict_lst=[]
    print_rank_0(f'*************medical eval*************')
    scores = dict()

    ctx = get_ctx(benchmark_config['device'])
    for test_data_path in benchmark_config['test_data_path']:
        print_rank_0(f'eval {test_data_path}...')
        with open(test_data_path,'r',encoding='utf-8') as f:
            for row in tqdm(f):
                line=json.loads(row)

                # run generation
                if 'data/test.json' == test_data_path:
                    prompt=line['question']
                    answer=line['response_chosen']
                else : # 'test_en_1.json' == eval_data_path:
                    prompt=line['instruction']+line['input']
                    answer=line['output']

                x=tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.special_tokens['<eos>']]
                x = (torch.tensor(x, dtype=torch.long, device=benchmark_config['device'])[None, ...])
                answer_list.append(answer)
                with torch.no_grad():
                    with ctx:
                        outputs = model.generate(x, max_new_tokens=benchmark_config['gen_params']['max_new_tokens'], 
                                                 temperature=benchmark_config['gen_params']['temperature'], 
                                                 top_k=benchmark_config['gen_params']['top_k'])
                        predict=tokenizer.decode(outputs[0])
                        predict_lst.append(predict)
                        # print_rank_0('\n---------------')
                        # print_rank_0('[prompt]:',prompt)
                        # print_rank_0('[answer]:',answer)
                        # print_rank_0('[predict]:',predict)
        
        target_lst=[jieba.lcut(result.lower()) for result in answer_list]
        preds_lst=[jieba.lcut(result.lower()) for result in predict_lst]
        score = compute_bleu(preds_lst, target_lst)*100
        print_rank_0(f'{test_data_path}: eval_scores: {score}')
        scores[test_data_path] = score
        logger.info(f'Model: {model_path_dir}. [medical] [{test_data_path}] scores: {score}')

    weighted_acc = sum(scores.values())/len(scores)
    logger.info(f'Model: {model_path_dir}. [medical] aver_scores: {weighted_acc}')


def eval_ceval(model, model_path_name, benchmark_dir, tokenizer, benchmark_config, logger):
    print_rank_0(f'*************CEval*************')
    from src.benchmark.ceval import CEval
    ceval = CEval(model, tokenizer, benchmark_dir, benchmark_config['device'])
    accs, average_acc=ceval.run(CEVAL_DATA_PATH, benchmark_config['gen_params']['shot'])
    for key in accs:
        logger.info(f'Model: {model_path_name}. [Ceval] [{key}] scores: {accs[key]*100}')
    logger.info(f'Model: {model_path_name}. [Ceval] aver_scores: {average_acc*100}')


def eval_mmlu(model, model_path_name, benchmark_dir, tokenizer, benchmark_config, logger):
    print_rank_0(f'*************MMLU*************')
    from src.benchmark.mmlu import mmlu_eval_func
    cat_cors, weighted_acc=mmlu_eval_func(MMLU_DATA_PATH, 
                                          benchmark_config['gen_params']['shot'], 
                                          model, tokenizer, model_path_name, benchmark_dir)
    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        logger.info(f'Model: {model_path_name}. [MMLU] [{cat}] scores: {cat_acc*100}')
    logger.info(f'Model: {model_path_name}. [MMLU] aver_scores: {weighted_acc*100}')


def eval_longbench(model, model_path_name, benchmark_dir, tokenizer, benchmark_config, logger):
    print_rank_0(f'*************LongBench*************')
    from src.benchmark.longbench import longbench_eval_func
    scores, weighted_acc=longbench_eval_func(LONGBENCH_DATA_PATH, benchmark_config, model, tokenizer, benchmark_dir)
    for key in scores:
        logger.info(f"Model: {model_path_name}. [LongBench] [{key}] '0-4k' scores: {scores[key]['0-4k']}")
        logger.info(f"Model: {model_path_name}. [LongBench] [{key}] '4-8k' scores: {scores[key]['4-8k']}")
        logger.info(f"Model: {model_path_name}. [LongBench] [{key}] '8k+'  scores: {scores[key]['8k+']}")
    logger.info(f"Model: {model_path_name}. [LongBench] 0-4k aver_scores: {weighted_acc['0-4k']}")
    logger.info(f"Model: {model_path_name}. [LongBench] 4-8k aver_scores: {weighted_acc['4-8k']}")
    logger.info(f"Model: {model_path_name}. [LongBench] 8k+  aver_scores: {weighted_acc['8k+']}")


def eval_LongEval(model, model_path_name, benchmark_dir, tokenizer, benchmark_config, logger):
    print_rank_0(f'*************LongEval*************')
    from src.benchmark.longeval import longeval_eval_func
    scores, weighted_acc=longeval_eval_func(LONGEVAL_DATA_PATH, model, tokenizer, benchmark_dir)
    for key in scores:
        logger.info(f'Model: {model_path_name}. [longEval] [{key}] scores: {scores[key]}')
    logger.info(f'Model: {model_path_name}. [longEval] aver_scores: {weighted_acc}')


def eval_perplexity(model, model_path_name, benchmark_dir, tokenizer, benchmark_config, logger):
    from src.benchmark.perplexity import Perplexity
    import pdb; pdb.set_trace()
    ppl_results, mean_perplexity = Perplexity.compute(PERPLEXITY_DATA_PATH, 
                                                      benchmark_config, 
                                                      model, 
                                                      tokenizer, 
                                                      benchmark_dir)
    for ppl_value in ppl_results:
        logger.info(f'Model: {model_path_name}. [longEval Perplexity] ppl: {ppl_value}')
    logger.info(f'Model: {model_path_name}. [longEval Perplexity] mean_ppl: {mean_perplexity}')


def eval_GSM8K(model, model_path_name, benchmark_dir, tokenizer, benchmark_config, logger):
    print_rank_0(f'*************GSM8K*************')
    from src.benchmark.gsm8k import GSM8K
    gsm8k = GSM8K(model, tokenizer, model_path_name)
    scores, weighted_acc=gsm8k.run(GSM8K_DATA_PATH, benchmark_dir, benchmark_config['gen_params']['shot'])
    for key in scores:
        logger.info(f'Model: {model_path_name}. [gsm8k] [{key}] scores: {scores[key]}')
    logger.info(f'Model: {model_path_name}. [gsm8k] aver_scores: {weighted_acc}')


def benchmark(model_path_dir, benchmark_config):
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    config_file = os.path.join(model_path_dir, "config.yaml")
    model_config = read_config(config_file)
    model,_ = init_model(model_config, model_path_dir)
    model = model.half().eval()
    model.to(benchmark_config['device'])
    if benchmark_config['compile']:
        print_rank_0("Compiling the model...")
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    benchmark_dir = model_path_dir + '_benchmark'
    os.makedirs(benchmark_dir, exist_ok=True)
    log_path = os.path.join(benchmark_dir, 'benchmark.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = get_logger(log_path)

    tokenizer=ChatGLMTokenizer()
    eval_medical(model, model_path_dir, tokenizer, benchmark_config, logger)
    eval_ceval(model, model_path_dir, benchmark_dir, tokenizer, benchmark_config, logger)
    eval_mmlu(model, model_path_dir, benchmark_dir, tokenizer, benchmark_config, logger)
    eval_longbench(model, model_path_dir, benchmark_dir, tokenizer, benchmark_config, logger)
    eval_LongEval(model, model_path_dir, benchmark_dir, tokenizer, benchmark_config, logger)
    eval_GSM8K(model, model_path_dir, benchmark_dir, tokenizer, benchmark_config, logger)
    # eval_perplexity(model, model_path_dir, benchmark_dir, tokenizer, benchmark_config, logger)


# I/O
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./out/pretrain_layer12_dim768_seq768', help="path to config")
    parser.add_argument("--benchmark_file", type=str, default='./config/train.yaml', help="path to config")
    args = parser.parse_args()
    
    model_path_dir = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path)
    benchmark_config = read_config(args.benchmark_file)

    print_rank_0(f'*************benchmark model: {model_path_dir}*************')
    benchmark(model_path_dir, benchmark_config)