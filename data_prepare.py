
from src.data.data_process import *
from src.data.sft_data_process import *
from src.utils import *

if __name__=="__main__":
    # 数据预处理-如果下载分词处理后的数据，可以不用执行以下函数
    print_rank_0('process wikipedia.')
    process_wiki_clean("wikipedia-cn-20230720-filtered.json")

    print_rank_0('process baidu.')
    process_baidu('563w_baidubaike.json')

    print_rank_0('process medical.')
    process_medical('medical_book_zh.json')
    process_medical('train_encyclopedia.json')

    print_rank_0('process c4.')
    process_c4('c4_zh')

    print_rank_0('process wudao.')
    process_wudao('WuDaoCorpus2.0_base_200G')

    print_rank_0('data processing finished!')
    
    # 分词处理后的文件列表
    collect_all_pretrain_data()

    print_rank_0('sft_to_pretrain.')
    # sft_to_pretrain()

    print_rank_0('sft_process.')
    sft_process()
