import json
import glob
import numpy as np
from tqdm import tqdm
from src.chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd
import os
from src.utils import check_is_processed
from datasets import load_dataset
from src.utils import get_logger


tokenizer = ChatGLMTokenizer()


BATCH_SIZE = 20000000
DATA_PATH = "./data"
PRETRAINED_DATA_PATH = "pretrain_data_bin"

GLOBAL_MIN_LEN = 10
logger = get_logger(os.path.join(DATA_PATH,'log.log'))

#from zhconv import convert
def process_wiki_clean(data_file):
    name = data_file.split('.')[0]
    data_dir_path = os.path.join(os.path.join(DATA_PATH, PRETRAINED_DATA_PATH), name)
    if check_is_processed(data_dir_path):
        print(f'[INFO] {data_file} has processed .')
        return
    
    data_path = os.path.join(DATA_PATH, data_file)
    if not os.path.exists(data_path):
        print(f'No exist data_file: {data_file}')
        print('download data from url: https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered.')
        # traindata = load_dataset("pleisto/wikipedia-cn-20230720-filtered", "wikipedia-cn-20230720-filtered", split="train")
        return
    else:
        with open(data_path,'r',encoding='utf-8') as f:
            traindata=json.load(f)

    batch_cnt = 0
    bin_name = f'{data_dir_path}/pretrain_{name}{batch_cnt}.bin'
    os.makedirs(os.path.dirname(bin_name), exist_ok=True)

    doc_ids=[]
    total_id_len = 0
    for line in tqdm(traindata):
        text=line['completion']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>GLOBAL_MIN_LEN:
            doc_ids+=text_id

        total_id_len += len(text_id)
        if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
            arr = np.array(doc_ids, dtype=np.uint16)
            with open(bin_name.replace('0.bin', f'{batch_cnt}.bin'), 'wb') as f2:
                f2.write(arr.tobytes())
            batch_cnt += 1
            doc_ids = []
            del arr

    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        with open(bin_name.replace('0.bin', f'{batch_cnt}.bin'), 'wb') as f2:
            f2.write(arr.tobytes())
        batch_cnt += 1

    print(f'processed {data_file} tokens: {total_id_len}')
    logger.info(f'processed {data_file} tokens: {total_id_len}')


def process_baidu(data_file):
    name = data_file.split('.')[0]
    data_dir_path = os.path.join(os.path.join(DATA_PATH, PRETRAINED_DATA_PATH), name)
    if check_is_processed(data_dir_path):
        print(f'[INFO] {data_file} has processed .')
        return
    
    data_path = os.path.join(DATA_PATH, data_file)
    if not os.path.exists(data_path):
        print(f'No exist data_file: {data_file}')
        print('download data from url: https://huggingface.co/datasets/xuqinyang/BaiduBaike-5.63M/blob/main/563w_baidubaike.json.')
        return
    
    batch_cnt=0
    doc_ids=[]
    total_id_len = 0
    bin_name = f'{data_dir_path}/pretrain_{name}{batch_cnt}.bin'
    os.makedirs(os.path.dirname(bin_name), exist_ok=True)

    f1=open(data_path,'r',encoding='utf-8')
    while True:
        line = f1.readline()
        if not line:
            break
        line=json.loads(line)
        text=''
        try:
            text+=line['title']+'：'+line['summary']
        except:
            pass
        for per in line['sections']:
            text+=per['title']+'：'+per['content']+'。'

        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id) > GLOBAL_MIN_LEN:
            doc_ids+=text_id

        total_id_len += len(text_id)
        if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
            arr = np.array(doc_ids, dtype=np.uint16)
            with open(bin_name.replace('0.bin', f'{batch_cnt}.bin'), 'wb') as f2:
                f2.write(arr.tobytes())
            batch_cnt += 1
            doc_ids = []
            del arr


    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        with open(bin_name.replace('0.bin', f'{batch_cnt}.bin'), 'wb') as f2:
            f2.write(arr.tobytes())
        batch_cnt += 1
    
    print(f'processed {data_file} tokens: {total_id_len}')
    logger.info(f'processed {data_file} tokens: {total_id_len}')



def process_medical(data_file):
    name = data_file.split('.')[0]
    data_dir_path = os.path.join(os.path.join(DATA_PATH, PRETRAINED_DATA_PATH), name)
    if check_is_processed(data_dir_path):
        print(f'[INFO] {data_file} has processed .')
        return
    
    data_path = os.path.join(DATA_PATH, data_file)
    if not os.path.exists(data_path):
        print(f'No exist data_file: {data_file}')
        print('download data from url: https://huggingface.co/datasets/shibing624/medical.')
        return
    
    batch_cnt=0
    doc_ids=[]
    total_id_len = 0
    bin_name = f'{data_dir_path}/pretrain_{name}{batch_cnt}.bin'
    os.makedirs(os.path.dirname(bin_name), exist_ok=True)

    f=open(data_path,'r',encoding='utf-8')
    while True:
        line=f.readline()
        if not line:
            break

        line=json.loads(line)
        text=line['text']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>GLOBAL_MIN_LEN:
            doc_ids+=text_id

        if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
            arr = np.array(doc_ids, dtype=np.uint16)
            with open(bin_name.replace('0.bin', f'{batch_cnt}.bin'), 'wb') as f2:
                f2.write(arr.tobytes())
            batch_cnt += 1
            doc_ids = []
            del arr
    
    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        with open(bin_name.replace('0.bin', f'{batch_cnt}.bin'), 'wb') as f2:
            f2.write(arr.tobytes())
        batch_cnt += 1
    
    print(f'processed {data_file} tokens: {total_id_len}')
    logger.info(f'processed {data_file} tokens: {total_id_len}')


def process_c4(data_file):
    name = data_file.split('.')[0]
    data_dir_path = os.path.join(os.path.join(DATA_PATH, PRETRAINED_DATA_PATH), name)
    if check_is_processed(data_dir_path):
        print(f'[INFO] {data_file} has processed .')
        return
    
    data_path = os.path.join(DATA_PATH, data_file)
    if not os.path.exists(data_path):
        print(f'No exist data_file: {data_file}')
        print('download data from url: https://github.com/DLLXW/baby-llama2-chinese.')
        return
    
    batch_cnt=0
    doc_ids=[]
    total_id_len = 0
    bin_name = f'{data_dir_path}/pretrain_{name}{batch_cnt}.bin'
    os.makedirs(os.path.dirname(bin_name), exist_ok=True)

    c4_zh_paths = glob.glob(data_path)
    c4_zh_paths=sorted(c4_zh_paths)
    print(len(c4_zh_paths))

    doc_ids=[]
    for per in tqdm(c4_zh_paths):
        with open(per,'r') as f:
            for line in f:
                text = json.loads(line)
                text = text['text']
                text_id=tokenizer.encode(text,add_special_tokens=False)
                text_id.append(tokenizer.special_tokens['<eos>'])
                if len(text_id) > GLOBAL_MIN_LEN:
                    doc_ids+=text_id
                
                if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
                    arr = np.array(doc_ids, dtype=np.uint16)
                    with open(bin_name.replace('0.bin', f'{batch_cnt}.bin'), 'wb') as f2:
                        f2.write(arr.tobytes())
                    batch_cnt += 1
                    doc_ids = []
                    del arr

    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        with open(bin_name.replace('0.bin', f'{batch_cnt}.bin'), 'wb') as f2:
            f2.write(arr.tobytes())
        batch_cnt += 1
    
    print(f'processed {data_file} tokens: {total_id_len}')
    logger.info(f'processed {data_file} tokens: {total_id_len}')



def process_wudao(data_file):
    name = data_file.split('.')[0]
    data_dir_path = os.path.join(os.path.join(DATA_PATH, PRETRAINED_DATA_PATH), name)
    if check_is_processed(data_dir_path):
        print(f'[INFO] {data_file} has processed .')
        return
    
    data_path = os.path.join(DATA_PATH, data_file)
    if not os.path.exists(data_path):
        print(f'No exist data_file: {data_file}')
        print('download data from url: https://data.baai.ac.cn/details/WuDaoCorporaText.')
        return
    
    batch_cnt=0
    doc_ids=[]
    total_id_len = 0
    bin_name = f'{data_dir_path}/pretrain_{name}{batch_cnt}.bin'
    os.makedirs(os.path.dirname(bin_name), exist_ok=True)

    wudao_zh_paths = glob.glob(data_path)
    wudao_zh_paths=sorted(wudao_zh_paths)
    print(len(wudao_zh_paths))#很多子文件

    doc_ids=[]
    for per in tqdm(wudao_zh_paths[320:]):#wudao_zh_paths[i:j]手动分片，一片片处理，不然太大一次性处理不完
        with open(per,'r') as f:
            data=json.load(f)
            for text in data:
                text = text['title'] + text['content']
                text_id=tokenizer.encode(text,add_special_tokens=False)
                text_id.append(tokenizer.special_tokens['<eos>'])
                if len(text_id) > GLOBAL_MIN_LEN:
                    doc_ids+=text_id

                if len(doc_ids) > 0 and len(doc_ids) > BATCH_SIZE:
                    arr = np.array(doc_ids, dtype=np.uint16)
                    with open(bin_name.replace('0.bin', f'{batch_cnt}.bin'), 'wb') as f2:
                        f2.write(arr.tobytes())
                    batch_cnt += 1
                    doc_ids = []
                    del arr

    if len(doc_ids) > 0:
        arr = np.array(doc_ids, dtype=np.uint16)
        with open(bin_name.replace('0.bin', f'{batch_cnt}.bin'), 'wb') as f2:
            f2.write(arr.tobytes())
        batch_cnt += 1
    
    print(f'processed {data_file} tokens: {total_id_len}')
    logger.info(f'processed {data_file} tokens: {total_id_len}')


def collect_all_pretrain_data():
    ### 将所有pretrain_xxx.bin的文件合并成pretrain_data.bin
    pretrain_data_bin_paths = []
    pretrain_data_dir = os.path.join(DATA_PATH, PRETRAINED_DATA_PATH)
    data_list = os.listdir(pretrain_data_dir)
    for data_dir in data_list:
        abs_data_dir = os.path.join(pretrain_data_dir, data_dir)
        bin_list = os.listdir(abs_data_dir)
        for bin_name in bin_list:
            if 'pretrain' in bin_name and bin_name.endswith('.bin'):
                pretrain_data_bin_paths.append(os.path.join(abs_data_dir, bin_name))

    if len(pretrain_data_bin_paths)==0:
        print(f'no bin data find in {pretrain_data_dir}')
        return

    print('concat pretrain_data.')
    data_lst = []
    for idx, data_file in enumerate(pretrain_data_bin_paths):
        print(f'[{idx}/{len(pretrain_data_bin_paths)}] read data: {data_file}')
        with open(data_file, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
            data_lst.append(data)

    arr = np.concatenate(data_lst)
    print(f'tokens: {arr.shape[0]/1024/1024} M')
    with open(os.path.join(DATA_PATH, 'pretrain_data.bin'), 'wb') as f:
        f.write(arr.tobytes())
    print('finished.')


def sft_to_pretrain():
    doc_ids=[]

    '''
    df=pd.read_csv('./data/medical_qa_144w.csv')
    for _,q,a in tqdm(df.itertuples()):
        q_id = tokenizer.encode(q,add_special_tokens=False)
        a_id = tokenizer.encode(a,add_special_tokens=False)
        #
        print(q)
        print(a)
        print('-----')
        text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
        if len(text_id) > GLOBAL_MIN_LEN:
            doc_ids+=text_id
    '''

    with open('./data/shibing624_medical/finetune/train_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id) > GLOBAL_MIN_LEN:
                doc_ids+=text_id
    with open('./data/shibing624_medical/finetune/test_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id) > GLOBAL_MIN_LEN:
                doc_ids+=text_id
    with open('./data/shibing624_medical/finetune/valid_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id) > GLOBAL_MIN_LEN:
                doc_ids+=text_id

    with open('./data/shibing624_medical/finetune/train_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id) > GLOBAL_MIN_LEN:
                doc_ids+=text_id
    with open('./data/shibing624_medical/finetune/test_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id) > GLOBAL_MIN_LEN:
                doc_ids+=text_id
    with open('./data/shibing624_medical/finetune/valid_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id) > GLOBAL_MIN_LEN:
                doc_ids+=text_id

    arr = np.array(doc_ids,dtype=np.uint16)
    print(arr.shape)
    with open('./data/medical_qa.bin','wb') as f:
        f.write(arr.tobytes())

    