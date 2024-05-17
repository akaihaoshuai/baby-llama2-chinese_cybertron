import json
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from src.data_prepare.data_process import *

if __name__=="__main__":
    # 数据预处理-如果下载分词处理后的数据，可以不用执行以下函数
    print('process wikipedia.')
    process_wiki_clean("wikipedia-cn-20230720-filtered.json")

    print('process baidu.')
    process_baidu('563w_baidubaike.json')

    print('process medical.')
    process_medical('medical_book_zh.json')
    process_medical('train_encyclopedia.json')

    print('process c4.')
    process_c4('c4_zh')

    print('process wudao.')
    process_wudao('WuDaoCorpus2.0_base_200G')

    print('data processing finished!')
    
    # 分词处理后的文件列表
    collect_all_pretrain_data()

    # print('sft_process.')
    # sft_to_pretrain()

