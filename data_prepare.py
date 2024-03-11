import json
import glob
import numpy as np
from tokenizer_model import *
import os
from src.data.data_process import *

BATCH_SIZE = 500000000
tokenizer = ChatGLMTokenizer(vocab_file='./tokenizer_model/chatglm_tokenizer/tokenizer.model')

def collect_pretrain_data(data_path, pretrain_data_list=None):
    ### 将所有pretrain_xxx.bin的文件合并成pretrain_data.bin
    if pretrain_data_list is None:
        pretrain_data_list = []
        file_list = os.listdir(data_path)
        for data_bin in file_list:
            if (not os.path.isdir(data_bin)) and 'pretrain' in data_bin and data_bin.endswith('.bin') and (not data_bin == 'pretrain_data.bin'):
                pretrain_data_list.append(os.path.join(data_path, data_bin))

    print('concat pretrain_data.')
    data_lst = []
    for idx, data_file in enumerate(pretrain_data_list):
        print(f'[{idx}/{len(pretrain_data_list)}] read data: {data_file}')
        with open(data_file, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
            data_lst.append(data)

    arr = np.concatenate(data_lst)
    print(f'tokens: {arr.shape[0]/1024/1024} M')
    with open(f'{data_path}/pretrain_data.bin', 'wb') as f:
        f.write(arr.tobytes())


def process_data_v0(tokenizer):
    save_all_text = False  # save_all_text_for_tokenizer

    print('process_baidu.')
    process_baidu('./data/563w_baidubaike.json', tokenizer, BATCH_SIZE, save_all_text)

    print('process_wiki_zh_clean.')
    process_wiki_zh_clean(tokenizer, save_all_text)

    print('process_medical: medical_book_zh')
    process_medical('./data/medical_book_zh.json', 'book',tokenizer, save_all_text)
    print('process_medical: train_encyclopedia')
    process_medical('./data/train_encyclopedia.json', 'encyclopedia',tokenizer, save_all_text)
    print('process_medical_qa.')
    process_medical_qa(tokenizer, save_all_text)

    collect_pretrain_data(GLOBAL_DATA_PATH)

    print('valid_medical.')
    process_valid_medical(tokenizer, save_all_text)

    if save_all_text:
        print('test_medical.')
        # 测试数据集不需要处理
        process_test_medical(tokenizer, save_all_text)

    print('sft_process.')
    sft_process(save_all_text)


def process_data_v1():
    save_all_text = False  # save_all_text_for_tokenizer

    # 中
    # print('process_CLUECorpusSmall.')
    # process_CLUECorpusSmall(tokenizer, BATCH_SIZE, save_all_text)
    #
    # print('process_wiki_zh_clean.')
    # process_wiki_zh_clean(tokenizer, save_all_text)
    #
    # print('process_baidu.')
    # process_baidu('./data/563w_baidubaike.json', tokenizer, BATCH_SIZE, save_all_text)
    #
    # # 这个数据集很大
    # print('process_mnbvc.')
    # process_MNBVC(tokenizer, BATCH_SIZE, save_all_text)
    #
    # print('process_Chinese-medical-dialogue-data.')
    # process_Chinese_medical_dialogue(tokenizer, BATCH_SIZE, save_all_text)
    #
    # # 英
    # # 这个数据集很大
    # print('process wikipedia.')
    # process_wiki(tokenizer, BATCH_SIZE, save_all_text)
    #
    # print('process_C4.')
    # process_C4(tokenizer, BATCH_SIZE, save_all_text)

    # 代码
    print('process github_code.')
    process_github_code(tokenizer, BATCH_SIZE, save_all_text)
    

    print('process_medical: medical_book_zh')
    process_medical('./data/medical_book_zh.json','book', tokenizer, save_all_text)
    print('process_medical: train_encyclopedia')
    process_medical('./data/train_encyclopedia.json','encyclopedia', tokenizer,save_all_text)
    print('process_medical_qa.')
    process_medical_qa(tokenizer, save_all_text)

    collect_pretrain_data(GLOBAL_DATA_PATH)

    print('valid_medical.')
    process_valid_medical(tokenizer,save_all_text)
    
    if save_all_text:
        print('test_medical.')
        # 测试数据集不需要处理
        process_test_medical(tokenizer, save_all_text)
    
    print('sft_process.')
    sft_process(save_all_text)
    sft_long_process(save_all_text)


if __name__=="__main__":

    # process_data_v0()
    process_data_v1()