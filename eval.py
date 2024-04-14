"""
Sample from the trained model with PyTorch
"""
import os
import json
from contextlib import nullcontext
import torch
import pandas as pd
from src.models.Jerry import Jerry
from tokenizer_model import ChatGLMTokenizer
import numpy as np
from setting import *
from src.utils import *
from src.share import *
from tqdm import tqdm

def compute_bleu(labels, preds, weights=None):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    weights = weights or (0.25, 0.25, 0.25, 0.25)
    return np.mean([sentence_bleu(references=[label],
                                  hypothesis=pred,
                                  smoothing_function=SmoothingFunction().method1,
                                  weights=weights) for label, pred in zip(labels, preds)])


def eval_medical(model,model_path_name,tokenizer,opt,ctx,logger):
    answer_list=[]
    predict_lst=[]
    print(f'*************medical eval*************')
    scores = dict()
    for eval_data_path in opt.test_data_path:
        print(f'eval {eval_data_path}...')
        with open(eval_data_path,'r',encoding='utf-8') as f:
            for row in tqdm(f):
                line=json.loads(row)

                # run generation
                if 'data/test.json' == eval_data_path:
                    prompt=line['question']
                    answer=line['response_chosen']
                else : # 'test_en_1.json' == eval_data_path:
                    prompt=line['instruction']+line['input']
                    answer=line['output']

                x=tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.special_tokens['<eos>']]
                x = (torch.tensor(x, dtype=torch.long, device=opt.device)[None, ...])
                answer_list.append(answer)
                with torch.no_grad():
                    with ctx:
                        outputs = model.generate(x, 2, opt.max_new_tokens, temperature=opt.temperature, top_k=opt.top_k)
                        predict=tokenizer.decode(outputs)
                        predict_lst.append(predict)
                        # print('\n---------------')
                        # print('[prompt]:',prompt)
                        # print('[answer]:',answer)
                        # print('[predict]:',predict)
        
        import jieba
        target_lst=[jieba.lcut(result.lower()) for result in answer_list]
        preds_lst=[jieba.lcut(result.lower()) for result in predict_lst]
        score = compute_bleu(preds_lst, target_lst)*100
        print(f'{eval_data_path}: eval_scores: {score}')
        scores[eval_data_path] = score
        logger.info(f'model: [medical] [{eval_data_path}] scores: {score}')

    weighted_acc = sum(scores.values())/len(scores)
    logger.info(f'model: {model_path_name}. [medical] aver_scores: {weighted_acc}')


def eval_ceval(model, model_path_name, tokenizer, opt, logger):
    print(f'*************CEval*************')
    from src.eval.ceval import CEval
    ceval = CEval(model, tokenizer, opt, model_path_name)
    accs, average_acc=ceval.run('data/ceval-exam',opt.shot)
    for key in accs:
        logger.info(f'model: {model_path_name}. [Ceval] [{key}] scores: {accs[key]*100}')
    logger.info(f'model: {model_path_name}. [Ceval] aver_scores: {average_acc*100}')


def eval_mmlu(model, model_path_name, tokenizer, opt, logger):
    print(f'*************MMLU*************')
    from src.eval.mmlu import mmlu_eval_func
    cat_cors, weighted_acc=mmlu_eval_func('data/mmlu', opt, model, tokenizer, model_path_name)
    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        logger.info(f'model: {model_path_name}. [MMLU] [{cat}] scores: {cat_acc*100}')
    logger.info(f'model: {model_path_name}. [MMLU] aver_scores: {weighted_acc*100}')


def eval_longbench(model, model_path_name, tokenizer, opt, logger):
    print(f'*************LongBench*************')
    from src.eval.longbench import longbench_eval_func
    scores, weighted_acc=longbench_eval_func('data/longBench', opt, model, tokenizer, model_path_name)
    for key in scores:
        logger.info(f"model: {model_path_name}. [LongBench] [{key}] '0-4k' scores: {scores[key]['0-4k']}")
        logger.info(f"model: {model_path_name}. [LongBench] [{key}] '4-8k' scores: {scores[key]['4-8k']}")
        logger.info(f"model: {model_path_name}. [LongBench] [{key}] '8k+'  scores: {scores[key]['8k+']}")
    logger.info(f"model: {model_path_name}. [LongBench] 0-4k aver_scores: {weighted_acc['0-4k']}")
    logger.info(f"model: {model_path_name}. [LongBench] 4-8k aver_scores: {weighted_acc['4-8k']}")
    logger.info(f"model: {model_path_name}. [LongBench] 8k+  aver_scores: {weighted_acc['8k+']}")


def eval_LongEval(model, model_path_name, tokenizer, opt, logger):
    print(f'*************LongEval*************')
    from src.eval.longeval import longeval_eval_func
    scores, weighted_acc=longeval_eval_func('data/longEval', opt, model, tokenizer, model_path_name)
    for key in scores:
        logger.info(f'model: {model_path_name}. [longEval] [{key}] scores: {scores[key]}')
    logger.info(f'model: {model_path_name}. [longEval] aver_scores: {weighted_acc}')


def eval_perplexity(model,model_path_name,tokenizer,opt,ctx,logger):
    from src.eval.perplexity import Perplexity
    ppl_results, mean_perplexity = Perplexity.compute('data/longEval', opt, model, tokenizer, model_path_name)
    for ppl_value in ppl_results:
        logger.info(f'model: {model_path_name}. [longEval Perplexity] ppl: {ppl_value}')
    logger.info(f'model: {model_path_name}. [longEval Perplexity] mean_ppl: {mean_perplexity}')


def eval_GSM8K(model, model_path_name, tokenizer, opt, logger):
    print(f'*************GSM8K*************')
    from src.eval.gsm8k import GSM8K
    gsm8k = GSM8K(model, tokenizer, opt, model_path_name)
    scores, weighted_acc=gsm8k.run(opt.shot, opt.split)
    for key in scores:
        logger.info(f'model: {model_path_name}. [gsm8k] [{key}] scores: {scores[key]}')
    logger.info(f'model: {model_path_name}. [gsm8k] aver_scores: {weighted_acc}')


def eval(model_path_name,opt,logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in opt.device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[opt.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast()

    model_path, state_dict, lora_path, lora_state_dict = read_ckpt(model_path_name)
    if state_dict is None:
        return
    
    opt.lora_path = lora_path
    model=init_model(opt)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    load_weight(model, state_dict, lora_state_dict, opt.merge_lora_on_load, strict=False)

    model=model.half().eval()
    model.to(opt.device)
    if opt.compile:
        print("Compiling the model...")
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # load the tokenizer
    tokenizer=ChatGLMTokenizer(vocab_file=opt.vocab_file)

    eval_medical(model, model_path_name, tokenizer, opt, ctx, logger)
    eval_ceval(model, model_path_name, tokenizer, opt, logger)
    eval_mmlu(model, model_path_name, tokenizer, opt, logger)
    eval_longbench(model, model_path_name, tokenizer, opt, logger)
    eval_LongEval(model, model_path_name, tokenizer, opt, logger)

    # eval_perplexity(model, model_path_name, tokenizer, opt, ctx, logger)
    # eval_GSM8K(model, model_path_name, tokenizer, opt, logger)


# I/O
if __name__=="__main__":
    # -----------------------------------------------------------------------------
    opt = get_parser_args()
    opt, _, _ = parser_all_config(opt)

    # start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    #exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------
    from src.share import get_logger
    log_dir = os.path.join(opt.out_dir, 'eval_all.log')
    if os.path.exists(log_dir):
        os.remove(log_dir) 
    logger = get_logger(log_dir)

    model_path_list = os.listdir(opt.out_dir)
    for model_path in model_path_list:
        model_path_ = os.path.join(opt.out_dir, model_path)

        if os.path.isdir(model_path_):
            opt.model_config = os.path.join(model_path_, 'config.yaml')

            opt, _ = parser_model_config(opt)

            model_list = os.listdir(model_path_)
            for model_name in model_list:
                eval_model_path_name = os.path.join(model_path_, model_name)
                
                if eval_model_path_name.endswith('.pth') or eval_model_path_name.endswith('.lora'):
                    print(f'*************eval model: {model_path_}*************')
                    eval(eval_model_path_name,opt,logger)