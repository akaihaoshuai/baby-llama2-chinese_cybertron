import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import gzip
from datasets import load_dataset
from src.share import get_logger
from src.utils import *

GLOBAL_DATA_PATH = './data'
GLOBAL_MIN_LEN = 15
logger = get_logger(os.path.join(GLOBAL_DATA_PATH,'log.log'))


def process_CLUECorpusSmall(tokenizer, BATCH_SIZE, save_all_text = False):
    data_path = GLOBAL_DATA_PATH

    if check_is_processed(f'{GLOBAL_DATA_PATH}/pretrain_CLUECorpusSmall', data_path):
        print(f'pretrain_CLUECorpusSmall has been processed')
        return

    dataset_list= {
        'comment2019zh__corpus',
        'news2016zh_corpus',
        'webText2019zh_corpus2',
        'wiki_zh'
    }
    data_dict = dict()
    for data_name in dataset_list:
        data_dict[data_name] = getAllFiles(os.path.join(data_path, data_name))

    for key in data_dict:
        print(f'process CLUECorpusSmall {key}')

        if save_all_text:
            corpus_txts = open(f'./data/tokenizer_CLUECorpusSmall_{key}.txt', 'w', encoding='utf-8')

        doc_ids = []
        batch_cnt = 0
        total_id_len = 0
        for data_list in tqdm(data_dict[key]):
            f1 = open(data_list, 'r', encoding='utf-8')
            while True:
                line = f1.readline()
                if not line:
                    break
                if len(line) < GLOBAL_MIN_LEN:
                    continue

                if not data_list.endswith('txt'):
                    line = json.loads(line)
                    line = line['text']

                if tokenizer is not None:
                    text_id = tokenizer.encode(line, add_special_tokens=False)
                    text_id.append(tokenizer.special_tokens['<eos>'])
                    if len(text_id) < GLOBAL_MIN_LEN:
                        continue

                    doc_ids += text_id
                    total_id_len += len(text_id)
                    
                if save_all_text:
                    corpus_txts.write(line)

                if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
                    arr = np.array(doc_ids, dtype=np.uint16)
                    with open(f'{data_path}/pretrain_CLUECorpusSmall_{key}_{batch_cnt}.bin', 'wb') as f:
                        f.write(arr.tobytes())
                    batch_cnt += 1
                    doc_ids = []
                    del arr

        if len(doc_ids) > 0:
            arr = np.array(doc_ids, dtype=np.uint16)
            with open(f'{data_path}/pretrain_CLUECorpusSmall_{key}_{batch_cnt}.bin', 'wb') as f:
                f.write(arr.tobytes())
        print(f'processed CLUECorpusSmall_{key} tokens: {total_id_len}')
        logger.info(f'processed CLUECorpusSmall_{key} tokens: {total_id_len}')

        if save_all_text:
            corpus_txts.close()


def process_C4(tokenizer, BATCH_SIZE, save_all_text=False):
    data_path = GLOBAL_DATA_PATH

    if check_is_processed(f'{GLOBAL_DATA_PATH}/pretrain_C4', data_path):
        print(f'pretrain_C4 has been processed')
        return

    dataset_list = {
        # 'en',
        # 'en.noblocklist',
        # 'en.noclean',
        'multilingual',
        'realnewslike',
    }
    data_dict = dict()
    for data_name in dataset_list:
        data_dict[data_name] = getAllFiles(os.path.join(data_path, "c4/"+data_name))

    def parse_gz(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield json.loads(l)

    total_id_len = 0
    for key in data_dict:
        print(f'process C4 {key}')

        if save_all_text:
            corpus_txts = open(f'./data/tokenizer_C4_{key}.txt', 'w', encoding='utf-8')

        doc_ids = []
        batch_cnt = 0
        total_id_len_per_type = 0
        for data_list in tqdm(data_dict[key]):
            for line in parse_gz(data_list):
                line = line['text']

                if not line:
                    break
                if len(line) < GLOBAL_MIN_LEN:
                    continue

                if tokenizer is not None:
                    text_id = tokenizer.encode(line, add_special_tokens=False)
                    text_id.append(tokenizer.special_tokens['<eos>'])
                    if len(text_id) < GLOBAL_MIN_LEN:
                        continue

                    doc_ids += text_id
                    total_id_len_per_type += len(text_id)

                if save_all_text:
                    corpus_txts.write(line)

                if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
                    arr = np.array(doc_ids, dtype=np.uint16)
                    with open(f'{data_path}/pretrain_C4_{key}_{batch_cnt}.bin', 'wb') as f:
                        f.write(arr.tobytes())
                    batch_cnt += 1
                    doc_ids = []
                    del arr

        if len(doc_ids) > 0:
            arr = np.array(doc_ids, dtype=np.uint16)
            with open(f'{data_path}/pretrain_C4_{key}_{batch_cnt}.bin', 'wb') as f:
                f.write(arr.tobytes())
        print(f'processed C4_{key} tokens: {total_id_len_per_type}')
        logger.info(f'processed C4_{key} tokens: {total_id_len_per_type}')
        total_id_len += total_id_len_per_type

        if save_all_text:
            corpus_txts.close()

    print(f'processed C4_{key} tokens: {total_id_len}')
    logger.info(f'processed C4_{key} tokens: {total_id_len}')


def process_github_code(tokenizer, BATCH_SIZE, save_all_text=False):
    # https://huggingface.co/datasets/codeparrot/github-code
    languages_list = {'C', 'C++', 'GO', 'Java', 'JavaScript', 'Python', 'Rust:'}

    data_path = GLOBAL_DATA_PATH
    for language_name in languages_list:
        batch_cnt = 0
        dateset_path = f'{data_path}/pretrain_githubcode_{language_name}_{batch_cnt}.bin'

        if os.path.exists(dateset_path):
            print(f'{dateset_path} has been processed')
            continue

        total_id_len = 0
        try:
            batch_cnt = 0
            print(f'load [github code] {language_name}')
            dataset = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=[language_name])
            doc_ids = []

            if save_all_text:
                corpus_txts = open(f'./data/tokenizer_githubcode_{language_name}.txt', 'w', encoding='utf-8')

            for line in tqdm(dataset):
                # next(iter(dataset))  # get the first line
                for paragraph in line['code']:
                    # rr=line['response_rejected']
                    if len(paragraph['内容']) < GLOBAL_MIN_LEN:
                        continue

                    if tokenizer is not None:
                        content_id = tokenizer.encode(paragraph['内容'], add_special_tokens=False)
                        # rr_id=tokenizer.encode(rr,add_special_tokens=False)
                        text_id = content_id + [tokenizer.special_tokens['<eos>']]
                        if len(text_id) > GLOBAL_MIN_LEN:
                            doc_ids += text_id
                            total_id_len += len(text_id)

                    if save_all_text:
                        corpus_txts.write(paragraph['内容'] + '\n')

                if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
                    arr = np.array(doc_ids, dtype=np.uint16)
                    target_p = dateset_path.replace(f'{0}.bin', f'{batch_cnt}.bin')
                    with open(target_p, 'wb') as f:
                        f.write(arr.tobytes())
                    batch_cnt += 1
                    doc_ids = []
                    del arr

            if len(doc_ids) > 0:
                arr = np.array(doc_ids, dtype=np.uint16)
                target_p = dateset_path.replace(f'{0}.bin', f'{batch_cnt}.bin')
                with open(target_p, 'wb') as f:
                    f.write(arr.tobytes())

            if save_all_text:
                corpus_txts.close()
        except:
            print(f'dowload {dateset_path} error....')

    print(f'processed process_MNBVC tokens: {total_id_len}')
    logger.info(f'processed process_MNBVC tokens: {total_id_len}')


def process_Chinese_medical_dialogue(tokenizer, BATCH_SIZE, save_all_text=False):
    data_path = GLOBAL_DATA_PATH

    if check_is_processed(f'{GLOBAL_DATA_PATH}/pretrain_Chinese_medical_dialogue', data_path):
        print(f'pretrain_Chinese_medical_dialogue has been processed')
        return

    if save_all_text:
        corpus_txts = open(f'./data/tokenizer_Chinese_medical_dialogue.txt', 'w', encoding='utf-8')

    data_files = getAllFiles(os.path.join(data_path, "Chinese-medical-dialogue-data/Data_数据"))

    doc_ids = []
    batch_cnt = 0
    total_id_len = 0
    for file_name in tqdm(data_files):
        if not file_name.endswith(".csv"):
            continue

        import chardet
        encoding = 'ISO'
        with open(file_name, 'r', encoding='gb2312') as f:
            line = f.readline()

            while True:
                try:
                    line = f.readline()
                except:
                    continue

                if not line:
                    break

                lines = line.split(',')
                if len(lines) != 4:
                    continue

                text = lines[1] +"。"+ lines[2]+"。"+lines[3]
                if len(text) < GLOBAL_MIN_LEN:
                    continue

                if tokenizer is not None:
                    text_id = tokenizer.encode(text, add_special_tokens=False)
                    text_id.append(tokenizer.special_tokens['<eos>'])
                    if len(text_id) < GLOBAL_MIN_LEN:
                        continue

                    doc_ids += text_id
                    total_id_len += len(text_id)

                if save_all_text:
                    corpus_txts.write(text)

                if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
                    arr = np.array(doc_ids, dtype=np.uint16)
                    with open(f'{data_path}/pretrain_Chinese_medical_dialogue_{batch_cnt}.bin', 'wb') as f:
                        f.write(arr.tobytes())
                    batch_cnt += 1
                    doc_ids = []
                    del arr

    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        with open(f'{data_path}/pretrain_Chinese_medical_dialogue_{batch_cnt}.bin', 'wb') as f:
            f.write(arr.tobytes())
    print(f'processed Chinese_medical_dialogue tokens: {total_id_len}')
    logger.info(f'processed Chinese_medical_dialogue tokens: {total_id_len}')

    if save_all_text:
        corpus_txts.close()
            
            
def process_baidu(filenam, tokenizer, BATCH_SIZE, save_all_text=False):
    batch_cnt = 0
    doc_ids = []
    data_path = GLOBAL_DATA_PATH

    if check_is_processed(f'{GLOBAL_DATA_PATH}/pretrain_baidubaike_563w', data_path):
        print(f'baidubaike_563w has been processed')
        return

    if save_all_text:
        corpus_txts = open(f'{data_path}/tokenizer_baidubaike_563w.txt', 'w', encoding='utf-8')

    if not os.path.exists(filenam):
        print(
            f'{filenam} is not exist. please download from: https://huggingface.co/datasets/xuqinyang/BaiduBaike-5.63M/blob/main/563w_baidubaike.json')
        return

    f1 = open(filenam, 'r', encoding='utf-8')
    total_id_len = 0
    while True:
        line = f1.readline()
        if not line:
            break
        line = json.loads(line)
        text = ''
        try:
            text += line['title'] + ': ' + line['summary']
        except:
            pass
        for per in line['sections']:
            text += per['title'] + ': ' + per['content'] + '。'

        if len(text) < GLOBAL_MIN_LEN:
            continue

        if tokenizer is not None:
            text_id = tokenizer.encode(text, add_special_tokens=False)
            text_id.append(tokenizer.special_tokens['<eos>'])
            doc_ids += text_id
            total_id_len+=len(text_id)

        if save_all_text:
            corpus_txts.write(text + '\n')

        if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
            arr = np.array(doc_ids, dtype=np.uint16)
            with open(f'{data_path}/pretrain_baidubaike_563w_{batch_cnt}.bin', 'wb') as f2:
                f2.write(arr.tobytes())
            batch_cnt += 1
            doc_ids = []
            del arr

    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        with open(f'{data_path}/pretrain_baidubaike_563w_{batch_cnt}.bin', 'wb') as f:
            f.write(arr.tobytes())

    if save_all_text:
        corpus_txts.close()

    print(f'processed baidubaike_563w tokens: {total_id_len}')
    logger.info(f'processed baidubaike_563w tokens: {total_id_len}')


# from zhconv import convert
def process_wiki_zh_clean(tokenizer, BATCH_SIZE, save_all_text=False):
    # https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered
    wiki_data_path = './data/wikipedia-cn-20230720-filtered/wikipedia-cn-20230720-filtered.json'
    data_path = GLOBAL_DATA_PATH
    if check_is_processed(f'{GLOBAL_DATA_PATH}/pretrain_wiki_zh_clean', data_path):
        print(f'pretrain_wiki_zh has been processed')
        return

    if not os.path.exists(wiki_data_path):
        print(
            f'{wiki_data_path} is not exist. please download from: https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered/blob/main/wikipedia-cn-20230720-filtered.json')
        return

    if save_all_text:
        corpus_txts = open(f'./data/tokenizer_wiki_zh_clean.txt', 'w', encoding='utf-8')

    with open(wiki_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    doc_ids = []
    total_id_len=0
    print(f'processed pretrain process_wiki_zh_clean')
    for line in tqdm(data):
        text = line['completion']
        if len(text) < GLOBAL_MIN_LEN:
            continue
            
        if tokenizer is not None:
            text_id = tokenizer.encode(text, add_special_tokens=False)
            text_id.append(tokenizer.special_tokens['<eos>'])
            doc_ids += text_id
            total_id_len += len(text_id)
        if save_all_text:
            corpus_txts.write(text + '\n')

    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        with open(f'{data_path}/pretrain_wiki_zh_clean.bin', 'wb') as f:
            f.write(arr.tobytes())

    if save_all_text:
        corpus_txts.close()

    print(f'processed pretrain wiki_zh_clean tokens: {total_id_len}')
    logger.info(f'processed pretrain wiki_zh_clean tokens: {total_id_len}')


def process_wiki(tokenizer, BATCH_SIZE, save_all_text=False):
    data_date = '20220301'
    data_type_list = ['simple']
    batch_cnt = 0
    data_path = GLOBAL_DATA_PATH

    total_id_len = 0
    for data_type in data_type_list:
        print(f'process wiki. date: {data_date}. type{data_type}.')
        dateset_path = f'{data_path}/pretrain_wikipedia_{data_date}_{data_type}_{batch_cnt}.bin'
        if os.path.exists(dateset_path):
            return

        if save_all_text:
            corpus_txts = open(f'./data/tokenizer_wikipedia.txt', 'w', encoding='utf-8')

        wiki_data = load_dataset("wikipedia", f"{data_date}.{data_type}")

        total_id_len_per_type = 0
        doc_ids = []
        for line in tqdm(wiki_data):
            for paragraph in line['段落']:
                # rr=line['response_rejected']
                if tokenizer is not None:
                    content_id = tokenizer.encode(paragraph['内容'], add_special_tokens=False)
                    # rr_id=tokenizer.encode(rr,add_special_tokens=False)
                    text_id = content_id + [tokenizer.special_tokens['<eos>']]
                    if len(text_id) > GLOBAL_MIN_LEN:
                        doc_ids += text_id
                        total_id_len_per_type += len(text_id)
                if save_all_text:
                    corpus_txts.write(paragraph['内容'] + '\n')


            if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
                arr = np.array(doc_ids, dtype=np.uint16)
                target_p = dateset_path.replace(f'{0}.bin', f'{batch_cnt}.bin')
                with open(target_p, 'wb') as f:
                    f.write(arr.tobytes())
                batch_cnt += 1
                doc_ids = []
                del arr

        if len(doc_ids) > 0:
            arr = np.array(doc_ids, dtype=np.uint16)
            target_p = dateset_path.replace(f'{0}.bin', f'{batch_cnt}.bin')
            with open(target_p, 'wb') as f:
                f.write(arr.tobytes())

        if save_all_text:
            corpus_txts.close()

        print(f'processed pretrain_wiki {data_date}.{data_type} tokens: {total_id_len_per_type}')
        logger.info(f'processed pretrain_wiki {data_date}.{data_type} tokens: {total_id_len_per_type}')
        total_id_len += total_id_len_per_type

    print(f'processed pretrain_wiki tokens: {total_id_len}')
    logger.info(f'processed pretrain_wiki tokens: {total_id_len}')


def process_wiki_en(tokenizer, BATCH_SIZE, save_all_text=False):
    data_path = GLOBAL_DATA_PATH
    batch_cnt = 0
    dateset_path = f'{data_path}/pretrain_tdtunlp_wikipedia_en_{batch_cnt}.bin'
    if os.path.exists(dateset_path):
        return

    if save_all_text:
        corpus_txts = open(f'./data/tokenizer_wikipedia_en.txt', 'w', encoding='utf-8')

    wiki_data = load_dataset("tdtunlp/wikipedia_en")
    doc_ids = []
    total_id_len = 0
    for paragraph in tqdm(wiki_data['train']):
        for line in paragraph['text'].split('\n'):
            # rr=line['response_rejected']
            if len(line) < GLOBAL_MIN_LEN:
                continue

            if tokenizer is not None:
                content_id = tokenizer.encode(line, add_special_tokens=False)
                # rr_id=tokenizer.encode(rr,add_special_tokens=False)
                text_id = content_id + [tokenizer.special_tokens['<eos>']]
                if len(text_id) < GLOBAL_MIN_LEN:
                    continue
                doc_ids += text_id
                total_id_len += len(text_id)

            if save_all_text:
                corpus_txts.write(line + '\n')


            if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
                arr = np.array(doc_ids, dtype=np.uint16)
                target_p = dateset_path.replace(f'{0}.bin', f'{batch_cnt}.bin')
                with open(target_p, 'wb') as f:
                    f.write(arr.tobytes())
                batch_cnt += 1
                doc_ids = []
                del arr

    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        target_p = dateset_path.replace(f'{0}.bin', f'{batch_cnt}.bin')
        with open(target_p, 'wb') as f:
            f.write(arr.tobytes())

    if save_all_text:
        corpus_txts.close()

    print(f'processed pretrain_tdtunlp_wikipedia_en tokens: {total_id_len}')
    logger.info(f'processed pretrain_tdtunlp_wikipedia_en tokens: {total_id_len}')


def process_MNBVC(tokenizer, BATCH_SIZE, save_all_text=False):
    # https://huggingface.co/datasets/liwu/MNBVC
    dataset_list = {
        'law_judgement',       # 法律
        'gov_xuexiqiangguo',   # 学习强国
        'gov_report',          # 政府工作报告
        'co_ann_report',       # 企业年报
        'code_metadata',       # 代码元数据
        'qa_zhihu',            # 知乎的问答
        'qa_wikihow',          # wikihow的问答  好像不存在??
        'qa_mfa:',             # 外交部问答数据
        'news_peoples_daily',  # 人民日报的文本
        # 'wikipedia',         # 维基百科的文本
        'qa_stackexchange',    # StackExchange的问答
        'qa_chatgpt',          # ChatGPT构造的问答语料
        'math_qa',             # 数学领域有关的问答
        'math_chat',           # 数学领域有关的对话数据数据，可以提升模型Chain of Thought的能力
        # 'crawler_oscar',     # 从CommonCrawl中清洗出来的通用文本数据，有C4了
    }

    data_path = GLOBAL_DATA_PATH
    total_id_len = 0
    for dataset_name in dataset_list:
        batch_cnt = 0
        dateset_path = f'{data_path}/pretrain_mnbvc_{dataset_name}_{batch_cnt}.bin'

        if os.path.exists(dateset_path):
            print(f'{dateset_path} has been processed')
            continue

        total_id_len_per_type = 0
        try:
            batch_cnt = 0
            print(f'load [MNBVC] {dataset_name}')
            dataset = load_dataset("liwu/MNBVC", dataset_name, split='train', streaming=True)
            doc_ids = []

            if save_all_text:
                corpus_txts = open(f'./data/tokenizer_MNBVC_{dataset_name}.txt', 'w', encoding='utf-8')

            if 'crawler_oscar' == dataset_name or 'gov_xuexiqiangguo'==dataset_name or 'news_peoples_daily'==dataset_name:
                for line in tqdm(dataset):
                    # next(iter(dataset))  # get the first line
                    for paragraph in line['段落']:
                        # rr=line['response_rejected']
                        if len(paragraph['内容']) < GLOBAL_MIN_LEN:
                            continue

                        if tokenizer is not None:
                            content_id = tokenizer.encode(paragraph['内容'], add_special_tokens=False)
                            # rr_id=tokenizer.encode(rr,add_special_tokens=False)
                            text_id = content_id + [tokenizer.special_tokens['<eos>']]
                            if len(text_id) > GLOBAL_MIN_LEN:
                                doc_ids += text_id
                                total_id_len_per_type += len(text_id)

                        if save_all_text:
                            corpus_txts.write(paragraph['内容'] + '\n')

                    if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
                        arr = np.array(doc_ids, dtype=np.uint16)
                        target_p = dateset_path.replace(f'{0}.bin', f'{batch_cnt}.bin')
                        with open(target_p, 'wb') as f:
                            f.write(arr.tobytes())
                        batch_cnt += 1
                        doc_ids = []
                        del arr
            elif 'qa_wikihow' == dataset_name or 'qa_chatgpt' == dataset_name or 'qa_zhihu' == dataset_name \
                    or 'math_qa' == dataset_name or 'qa_stackexchange' == dataset_name or 'law_judgement' == dataset_name\
                    or 'math_chat' == dataset_name or 'code_metadata' == dataset_name or 'gov_report' == dataset_name \
                    or 'co_ann_report' == dataset_name:
                for line in tqdm(dataset):
                    # next(iter(dataset))  # get the first line
                    if 'text' in line:
                        text = line['text']
                    elif '主题' in line:
                        text = line['主题']
                        for response in line['回复']:
                            text += response['回复']
                    else:
                        text = line['问'] + '。 ' + line['答']

                    if len(text) < GLOBAL_MIN_LEN:
                        continue

                    # rr=line['response_rejected']
                    if tokenizer is not None:
                        text_id = tokenizer.encode(text, add_special_tokens=False)
                        text_id = text_id + [tokenizer.special_tokens['<eos>']]
                        if len(text_id) < GLOBAL_MIN_LEN:
                            continue
                        doc_ids += text_id
                        total_id_len_per_type += len(text_id)

                    if save_all_text:
                        corpus_txts.write(text + '\n')

                    if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
                        arr = np.array(doc_ids, dtype=np.uint16)
                        target_p = dateset_path.replace(f'{0}.bin', f'{batch_cnt}.bin')
                        with open(target_p, 'wb') as f:
                            f.write(arr.tobytes())
                        batch_cnt += 1
                        doc_ids = []
                        del arr
            else:
                print(f'read {dateset_path} format error.....')


            if len(doc_ids) > 0:
                arr = np.array(doc_ids, dtype=np.uint16)
                target_p = dateset_path.replace(f'{0}.bin', f'{batch_cnt}.bin')
                with open(target_p, 'wb') as f:
                    f.write(arr.tobytes())

            if save_all_text:
                corpus_txts.close()
        except:
            print(f'download {dateset_path} error.....')

        print(f'process MNBVC {dateset_path} tokens: {total_id_len_per_type}')
        logger.info(f'process MNBVC {dateset_path} tokens: {total_id_len_per_type}')
        total_id_len+=total_id_len_per_type

    print(f'processed MNBVC all tokens: {total_id_len}')
    logger.info(f'processed MNBVC all tokens: {total_id_len}')

def process_medical(file_path, name, tokenizer, save_all_text=False):
    data_path = GLOBAL_DATA_PATH
    if check_is_processed(f'{data_path}/pretrain_medical_{name}.bin',data_path):
        print(f'pretrain_medical has been processed')
        return

    if not os.path.exists(file_path):
        print(
            f'{file_path} is not exist. please download from: https://huggingface.co/datasets/shibing624/medical/blob/main/pretrain/medical_book_zh.json')
        return

    if save_all_text:
        corpus_txts = open(f'{GLOBAL_DATA_PATH}//tokenizer_medical_{name}.txt', 'w', encoding='utf-8')

    f = open(file_path, 'r', encoding='utf-8')
    doc_ids = []
    total_id_len = 0
    while True:
        line = f.readline()
        if not line:
            break
        line = json.loads(line)
        text = line['text']
        
        if tokenizer is not None:
            text_id = tokenizer.encode(text, add_special_tokens=False)
            text_id.append(tokenizer.special_tokens['<eos>'])
            if len(text_id) > GLOBAL_MIN_LEN:
                doc_ids += text_id
                total_id_len += len(text_id)

        if save_all_text:
            corpus_txts.write(text + '\n')

    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        with open(f'{data_path}/pretrain_medical_{name}.bin', 'wb') as f:
            f.write(arr.tobytes())

    if save_all_text:
        corpus_txts.close()

    print(f'processed process_medical tokens: {total_id_len}')
    logger.info(f'processed process_medical tokens: {total_id_len}')


def process_medical_qa(tokenizer, save_all_text=False):
    data_path = GLOBAL_DATA_PATH
    if check_is_processed(f'{GLOBAL_DATA_PATH}/pretrain_medical_qa.bin',data_path):
        print(f'pretrain_medical_qa has been processed')
        return

    doc_ids = []

    if save_all_text:
        corpus_txts = open(f'./data/tokenizer_medical_qa.txt', 'w', encoding='utf-8')

    print('process_medical_qa: train')
    data_name = f'{data_path}/train.json'
    total_id_len = 0
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/shibing624/medical')
    else:
        with open(data_name, 'r', encoding='utf-8') as f:
            for row in tqdm(f):
                line = json.loads(row)
                q = line['question']
                rc = line['response_chosen']
                # rr=line['response_rejected']
                
                if tokenizer is not None:
                    q_id = tokenizer.encode(q, add_special_tokens=False)
                    rc_id = tokenizer.encode(rc, add_special_tokens=False)
                    # rr_id=tokenizer.encode(rr,add_special_tokens=False)
                    text_id = q_id + rc_id + [tokenizer.special_tokens['<eos>']]
                    if len(text_id) > GLOBAL_MIN_LEN:
                        doc_ids += text_id
                        total_id_len += len(text_id)

                if save_all_text:
                    corpus_txts.write(q + rc + '\n')

    print('process_medical_qa: train_en_1')
    data_name = f'{data_path}/train_en_1.json'
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/shibing624/medical')
    else:
        with open(data_name, 'r', encoding='utf-8') as f:
            for row in tqdm(f):
                line = json.loads(row)
                q = line['input']
                a = line['output']

                if tokenizer is not None:
                    q_id = tokenizer.encode(q, add_special_tokens=False)
                    a_id = tokenizer.encode(a, add_special_tokens=False)
                    text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
                    if len(text_id) > GLOBAL_MIN_LEN:
                        doc_ids += text_id
                        total_id_len += len(text_id)
                if save_all_text:
                    corpus_txts.write(q + a + '\n')

    print('process_medical_qa: train_zh_0')
    data_name = f'{data_path}/train_zh_0.json'
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/shibing624/medical')
    else:
        with open(f'{data_path}/train_zh_0.json', 'r', encoding='utf-8') as f:
            for row in tqdm(f):
                line = json.loads(row)
                q = line['instruction'] + line['input']
                a = line['output']
                if tokenizer is not None:
                    q_id = tokenizer.encode(q, add_special_tokens=False)
                    a_id = tokenizer.encode(a, add_special_tokens=False)
                    text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
                    if len(text_id) > GLOBAL_MIN_LEN:
                        doc_ids += text_id
                        total_id_len += len(text_id)
                if save_all_text:
                    corpus_txts.write(q + a + '\n')

    print('process_medical_qa: train_encyclopedia')
    data_name = f'{data_path}/train_encyclopedia.json'
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/shibing624/medical')
    else:
        with open(data_name, 'r', encoding='utf-8') as f:
            for row in tqdm(f):
                line = json.loads(row)
                text = line['text']
                if tokenizer is not None:
                    text_id = tokenizer.encode(text, add_special_tokens=False)
                    text_id.append(tokenizer.special_tokens['<eos>'])
                    if len(text_id) > GLOBAL_MIN_LEN:
                        doc_ids += text_id
                        total_id_len += len(text_id)
                if save_all_text:
                    corpus_txts.write(text+ '\n')

    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        print(arr.shape)
        with open(f'{data_path}/pretrain_medical_qa.bin', 'wb') as f:
            f.write(arr.tobytes())

    if save_all_text:
        corpus_txts.close()

    print(f'processed pretrain_medical_qa tokens: {total_id_len}')
    logger.info(f'processed pretrain_medical_qa tokens: {total_id_len}')


def process_valid_medical(tokenizer, save_all_text=False):
    data_path = GLOBAL_DATA_PATH
    if check_is_processed(f'{data_path}/valid_data.bin',data_path):
        print(f'valid_data has been processed')
        return

    doc_ids = []
    total_id_len = 0

    if save_all_text:
        corpus_txts = open(f'{data_path}/tokenizer_valid_medical.txt', 'w', encoding='utf-8')

    print('valid_medical: valid')
    data_name = f'{data_path}/valid.json'
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/shibing624/medical')
    else:
        with open(data_name, 'r', encoding='utf-8') as f:
            for row in tqdm(f):
                line = json.loads(row)
                q = line['question']
                rc = line['response_chosen']
                rr = line['response_rejected']
                if tokenizer is not None:
                    q_id = tokenizer.encode(q, add_special_tokens=False)
                    rc_id = tokenizer.encode(rc, add_special_tokens=False)
                    rr_id = tokenizer.encode(rr, add_special_tokens=False)
                    text_id = q_id + rc_id + rr_id + [tokenizer.special_tokens['<eos>']]
                    if len(text_id) > GLOBAL_MIN_LEN:
                        doc_ids += text_id
                        total_id_len += len(text_id)
                if save_all_text:
                    corpus_txts.write(q + rc + '\n')

    print('valid_medical: valid_en_1')
    data_name = f'{data_path}/valid_en_1.json'
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/shibing624/medical')
    else:
        with open(data_name, 'r', encoding='utf-8') as f:
            for row in tqdm(f):
                line = json.loads(row)
                q = line['input']
                a = line['output']
                if tokenizer is not None:
                    q_id = tokenizer.encode(q, add_special_tokens=False)
                    a_id = tokenizer.encode(a, add_special_tokens=False)
                    text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
                    if len(text_id) > GLOBAL_MIN_LEN:
                        doc_ids += text_id
                        total_id_len += len(text_id)
                if save_all_text:
                    corpus_txts.write(q +  a + '\n')

    print('valid_medical: valid_zh_0')
    data_name = f'{data_path}/valid_zh_0.json'
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/shibing624/medical')
    else:
        with open(data_name, 'r', encoding='utf-8') as f:
            for row in tqdm(f):
                line = json.loads(row)
                q = line['instruction'] + line['input']
                a = line['output']
                if tokenizer is not None:
                    q_id = tokenizer.encode(q, add_special_tokens=False)
                    a_id = tokenizer.encode(a, add_special_tokens=False)
                    text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
                    if len(text_id) > GLOBAL_MIN_LEN:
                        doc_ids += text_id
                        total_id_len += len(text_id)
                if save_all_text:
                    corpus_txts.write(q +  a + '\n')

    print('valid_medical: valid_encyclopedia')
    data_name = f'{data_path}/shibing624-medical-pretrain/pretrain/valid_encyclopedia.json'
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/shibing624/medical')
    else:
        with open(data_name, 'r', encoding='utf-8') as f:
            for row in tqdm(f):
                line = json.loads(row)
                text = line['text']
                if tokenizer is not None:
                    text_id = tokenizer.encode(text, add_special_tokens=False)
                    text_id.append(tokenizer.special_tokens['<eos>'])
                    if len(text_id) > GLOBAL_MIN_LEN:
                        doc_ids += text_id
                        total_id_len += len(text_id)
                if save_all_text:
                    corpus_txts.write(text + '\n')

    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        print(arr.shape)
        with open(f'{data_path}/valid_data.bin', 'wb') as f:
            f.write(arr.tobytes())

    if save_all_text:
        corpus_txts.close()

    print(f'processed process_valid_medical tokens: {total_id_len}')
    logger.info(f'processed process_valid_medical tokens: {total_id_len}')


def process_test_medical(tokenizer, save_all_text=False):
    data_path = GLOBAL_DATA_PATH
    if check_is_processed(f'{GLOBAL_DATA_PATH}/test_data.bin',data_path):
        print(f'test_data has been processed')
        return

    doc_ids = []
    total_id_len = 0

    if save_all_text:
        corpus_txts = open(f'./data/tokenizer_test_medical.txt', 'w', encoding='utf-8')

    print('test_medical: test_en_1')
    data_name = f'{data_path}/test_en_1.json'
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/shibing624/medical')
    else:
        with open(data_name, 'r', encoding='utf-8') as f:
            for row in tqdm(f):
                line = json.loads(row)
                q = line['input']
                a = line['output']
                if tokenizer is not None:
                    q_id = tokenizer.encode(q, add_special_tokens=False)
                    a_id = tokenizer.encode(a, add_special_tokens=False)
                    text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
                    if len(text_id) > GLOBAL_MIN_LEN:
                        doc_ids += text_id
                        total_id_len += len(text_id)
                if save_all_text:
                    corpus_txts.write(q + a +'\n')

    print('test_medical: test_zh_0')
    data_name = f'{data_path}/test_zh_0.json'
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/shibing624/medical')
    else:
        with open(data_name, 'r', encoding='utf-8') as f:
            for row in tqdm(f):
                line = json.loads(row)
                q = line['instruction'] + line['input']
                a = line['output']
                if tokenizer is not None:
                    q_id = tokenizer.encode(q, add_special_tokens=False)
                    a_id = tokenizer.encode(a, add_special_tokens=False)
                    text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
                    if len(text_id) > GLOBAL_MIN_LEN:
                        doc_ids += text_id
                        total_id_len += len(text_id)
                if save_all_text:
                    corpus_txts.write(q + a +'\n')

    print('test_medical: test_encyclopedia')
    data_name = f'{data_path}/test_encyclopedia.json'
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/shibing624/medical')
    else:
        with open(data_name, 'r', encoding='utf-8') as f:
            for row in tqdm(f):
                line = json.loads(row)
                q = line['instruction'] + line['input']
                a = line['output']
                if tokenizer is not None:
                    q_id = tokenizer.encode(q, add_special_tokens=False)
                    a_id = tokenizer.encode(a, add_special_tokens=False)
                    text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
                    if len(text_id) > GLOBAL_MIN_LEN:
                        doc_ids += text_id
                        total_id_len += len(text_id)
                if save_all_text:
                    corpus_txts.write(q + a +'\n')

    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        print(arr.shape)
        with open(f'{data_path}/test_data.bin', 'wb') as f:
            f.write(arr.tobytes())

    if save_all_text:
        corpus_txts.close()

    print(f'processed process_test_medical tokens: {total_id_len}')
    logger.info(f'processed process_test_medical tokens: {total_id_len}')


def sft_process(save_all_text=False):
    data_path = GLOBAL_DATA_PATH
    if check_is_processed(f'{GLOBAL_DATA_PATH}/sft_data.csv',data_path):
        print(f'sft_data has been processed')
        return

    if save_all_text:
        corpus_txts = open(f'./data/tokenizer_sft_alpaca_gpt4_data_zh.txt', 'w', encoding='utf-8')

    data_name = f'{data_path}/alpaca-gpt4-data-zh/alpaca_gpt4_data_zh.json'
    if not os.path.exists(data_name):
        print(
            f'{data_name} is not exist. please download from: https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh')
    else:
        with open(data_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

        q_lst = []
        a_lst = []
        for per in tqdm(data):
            q = per['instruction'] + per['input']
            a = per['output']
            if len(q) < 10 or len(a) < 5 or len(q) > 256 or len(a) > 256:
                continue
            q_lst.append(q)
            a_lst.append(a)

            if save_all_text:
                corpus_txts.write(q + a + '\n')


    data_name = f'{data_path}/Belle_open_source_1M.json'
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/BelleGroup/train_1M_CN')
    else:
        f = open(data_name, 'r', encoding='utf-8')
        while True:
            line = f.readline()
            if not line:
                break
            per = json.loads(line)
            q = per['instruction'] + per['input']
            a = per['output']
            if len(q) < 10 or len(a) < 5 or len(q) > 256 or len(a) > 256:
                continue
            q_lst.append(q)
            a_lst.append(a)

            if save_all_text:
                corpus_txts.write(q + a + '\n')

    data_name = f'{data_path}/moss-003-sft-no-tools.jsonl'
    if not os.path.exists(data_name):
        print(f'{data_name} is not exist. please download from: https://huggingface.co/datasets/fnlp/moss-003-sft-data')
    else:
        f = open(data_name, 'r', encoding='utf-8')
        while True:
            line = f.readline()
            if not line:
                break
            per = json.loads(line)
            q = per['instruction'] + per['input']
            a = per['output']
            if len(q) < 10 or len(a) < 5 or len(q) > 256 or len(a) > 256:
                continue
            q_lst.append(q)
            a_lst.append(a)

            if save_all_text:
                corpus_txts.write(q + a + '\n')

    df = pd.DataFrame(columns=['prompt', 'answer'])
    df['prompt'] = q_lst
    df['answer'] = a_lst
    df.to_csv('data/sft_data.csv', index=False)
    print(df)

    if save_all_text:
        corpus_txts.close()


def sft_long_process(save_all_text=False):
    data_path = GLOBAL_DATA_PATH
    if check_is_processed(f'{GLOBAL_DATA_PATH}/sft_long_data_train.csv',data_path):
        print(f'sft_data has been processed')
        return

    if save_all_text:
        corpus_txts = open(f'./data/tokenizer_sft_long_RedPajama-Data-1T-Sample.txt', 'w', encoding='utf-8')

    data_path = "../../datasets/RedPajama-Data-1T-Sample"
    if not os.path.exists(data_path):
        print(
            f'RedPajama-Data-1T-Sample is not exist. please download from: https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh')
    else:
        dataset = load_dataset("../../datasets/RedPajama-Data-1T-Sample", cache_dir='./cache')
        for line in tqdm(dataset["train"]):
            q_lst = []
            a_lst = []
            for per in tqdm(line):
                q = per['instruction'] + per['input']
                a = per['output']
                if len(q) < 10 or len(a) < 5 or len(q) > 256 or len(a) > 256:
                    continue
                q_lst.append(q)
                a_lst.append(a)

                if save_all_text:
                    corpus_txts.write(q + a + '\n')

        df = pd.DataFrame(columns=['prompt', 'answer'])
        df['prompt'] = q_lst
        df['answer'] = a_lst
        df.to_csv('data/sft_long_data_train.csv', index=False)
        for line in tqdm(dataset["val"]):
            q_lst = []
            a_lst = []
            for per in tqdm(line):
                q = per['instruction'] + per['input']
                a = per['output']
                if len(q) < 10 or len(a) < 5 or len(q) > 256 or len(a) > 256:
                    continue
                q_lst.append(q)
                a_lst.append(a)

                if save_all_text:
                    corpus_txts.write(q + a + '\n')

        df = pd.DataFrame(columns=['prompt', 'answer'])
        df['prompt'] = q_lst
        df['answer'] = a_lst
        df.to_csv('data/sft_long_data_val.csv', index=False)

    if save_all_text:
        corpus_txts.close()