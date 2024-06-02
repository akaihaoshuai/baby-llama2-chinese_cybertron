import glob
import os

from tqdm import  tqdm
from src.utils import *

def collect_data(merge_txt_list,merged_txt_path):
    corpus=open(merged_txt_path,'w',encoding='utf-8')
    cnt=0
    for idx, file in enumerate(merge_txt_list):
        print_rank_0(f'[{idx}/{len(merge_txt_list)}] open {file}...')
        with open(file,'r',encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                # print_rank_0(line.strip())
                if len(line.strip())>50:
                    corpus.write(line.strip()+'\n')
                    cnt+=1
    print_rank_0(cnt)
    # 9853042
    corpus.close()

def train_tokenizer(merged_txt_path,tokenizer_name):
    import time
    import sentencepiece as spm

    """
    sentencepiece 参数
    trainer_spec {
      input: data/corpus.txt
      input_format: #
      model_prefix: open_llama # 模型输出路径
      model_type: BPE # 模型类型 bpe、char、word、unigram(gram)
      vocab_size: 50000 # 词汇表大小，数量越大训练越慢，太小(<4000)可能训练不了
      self_test_sample_size: 0
      character_coverage: 0.9995 # 模型中覆盖的字符数
      input_sentence_size: 0
      shuffle_input_sentence: 0
      seed_sentencepiece_size: 1000000 # 
      shrinking_factor: 0.75
      max_sentence_length: 16384 # 最大句子长度,默认是4192,长度按照字节计算,一个中文代表长度为2
      num_threads: 16 # 进程个数
      num_sub_iterations: 2
      max_sentencepiece_length: 16
      split_by_unicode_script: 1
      split_by_number: 1
      split_by_whitespace: 1
      split_digits: 1
      pretokenization_delimiter: 
      treat_whitespace_as_suffix: 0
      allow_whitespace_only_pieces: 1
      required_chars: 
      byte_fallback: 1
      vocabulary_output_piece_score: 1
      train_extremely_large_corpus: 1
      hard_vocab_limit: 1
      use_all_vocab: 0 # 使用
      unk_id: 0
      bos_id: 1
      eos_id: 2
      pad_id: 3
      unk_piece: <unk>
      bos_piece: <s>
      eos_piece: </s>
      pad_piece: <pad>
      unk_surface:  ⁇ 
      enable_differential_privacy: 0
      differential_privacy_noise_level: 0
      differential_privacy_clipping_threshold: 0
    }
    normalizer_spec {
      name: nfkc
      add_dummy_prefix: 1
      remove_extra_whitespaces: 0
      escape_whitespaces: 1
      normalization_rule_tsv: 
    }
    """
    print_rank_0(f'[tokenizer] train...')

    start_time = time.time()
    spm.SentencePieceTrainer.train(
        input=merged_txt_path,  # 输入文件
        model_prefix=tokenizer_name,  # 模型前缀
        shuffle_input_sentence=False,  # 是否打乱句子
        train_extremely_large_corpus=True,
        # hyperparameters of tokenizer
        max_sentence_length=16384,  # 句子最大长度
        pad_id=3,
        model_type="BPE",
        vocab_size=50000,
        split_digits=True,
        split_by_unicode_script=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nfkc",
    )

    end_time = time.time()
    print_rank_0(f'[tokenizer] train time: {end_time - start_time}')

    print_rank_0(f'[tokenizer] test...')
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(f'{tokenizer_name}.model')
    text = """
    垃圾分类，一般是指按一定规定或标准将垃圾分类储存、投放和搬运，从而转变成公共资源的一系列活动的总称。
    """

    pieces_list = sp_model.encode_as_pieces(text)
    ids_list = sp_model.encode_as_ids(text)
    # encode: text => id
    print_rank_0(pieces_list)
    print_rank_0(ids_list)

    # decode: id => text
    # print_rank_0(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est', 'ly']))
    # print_rank_0(sp.decode_ids([209, 31, 9, 375, 586, 34]))
    print_rank_0(sp_model.decode_pieces([pieces_list]))
    print_rank_0(sp_model.decode_ids([ids_list]))


def merge_tokenizers(llama_tokenizer, chinese_sp_model):
    from transformers import LlamaTokenizer
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
    # load
    # llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)  # 原生LLaMA分词模型
    # chinese_sp_model = spm.SentencePieceProcessor()
    # chinese_sp_model.Load(chinese_sp_model_file)

    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

    # print number of tokens
    print_rank_0(len(llama_tokenizer),len(chinese_sp_model))
    print_rank_0(llama_tokenizer.all_special_tokens)
    print_rank_0(llama_tokenizer.all_special_ids)
    print_rank_0(llama_tokenizer.special_tokens_map)

    ## Add Chinese tokens to LLaMA tokenizer
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
    print_rank_0(len(llama_spm_tokens_set))
    print_rank_0(f"Before:{len(llama_spm_tokens_set)}")
    for p in chinese_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)  # 将训练的分词模型追加新的token到之前的模型
    print_rank_0(f"New model pieces: {len(llama_spm.pieces)}")

    ## Save
    model_file = 'tokenizer.model'
    output_sp_dir = 'merged_tokenizer_sp'
    output_hf_dir = 'merged_tokenizer_hf'  # the path to save Chinese-LLaMA tokenizer
    os.makedirs(output_sp_dir, exist_ok=True)
    with open(output_sp_dir + f'/{model_file}', 'wb') as f:
        f.write(llama_spm.SerializeToString())

    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + f'/{model_file}')
    tokenizer.save_pretrained(output_hf_dir)
    print_rank_0(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")

    # Test
    # llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    merged_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
    print_rank_0(merged_tokenizer.all_special_tokens)
    print_rank_0(merged_tokenizer.all_special_ids)
    print_rank_0(merged_tokenizer.special_tokens_map)
    text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。'''
    text = '''大模型是指具有非常大的参数数量的人工神经网络模型。 在深度学习领域，大模型通常是指具有数亿到数万亿参数的模型。'''
    print_rank_0("Test text:\n", text)
    print_rank_0(f"Tokenized by LLaMA tokenizer:{len(llama_tokenizer.tokenize(text))},{llama_tokenizer.tokenize(text)}")
    print_rank_0(f"Tokenized by merged tokenizer:{len(merged_tokenizer.tokenize(text))},{merged_tokenizer.tokenize(text)}")

def eval_tokenizer(tokenizer,txt_file_path,res_csv_path):
    # tokenizer = LlamaTokenizer.from_pretrained('merged_tokenizer_hf_60k')
    # llama_tokenizer = LlamaTokenizer.from_pretrained('llama')
    print_rank_0(f'[tokenizer] eval...')

    print_rank_0(tokenizer)

    num_tokens = []
    num_ids = []
    num_ids_llama = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            encode = tokenizer(line)
            # print_rank_0(encode)
            # print_rank_0(len(line),len(encode['input_ids']))
            num_tokens.append(len(line))
            num_ids.append(len(encode['input_ids']))

    import pandas as pd
    df = pd.DataFrame({'num_tokens': num_tokens, 'num_ids': num_ids})
    df = df.sort_values(by=["num_tokens"], ascending=True)
    df.to_csv(res_csv_path, index=False)

    data = pd.read_csv(res_csv_path)
    data['raw_ratio'] = data['num_tokens'] / data['num_ids']
    print_rank_0(data.describe(percentiles=[0.3, 0.8]))
    data.describe(percentiles=[0.3, 0.8]).to_excel(res_csv_path.replace('.csv', '.xlsx'),index=True)


if __name__=="__main__":
    # 可以不重新训练tokenizer
    # 参考: https://github.com/yanqiangmiffy/how-to-train-tokenizer
    data_path = './data'
    merge_txt_list = []
    for file in os.listdir(data_path):
        if file.endswith('txt') and 'tokenizer_' in file:
            merge_txt_list.append(os.path.join(data_path, file))

    merged_txt_path = os.path.join(data_path,'merged_tokenizer.txt')
    # collect_data(merge_txt_list,merged_txt_path)

    tokenizer_name = 'baby40k'
    train_tokenizer(merged_txt_path, tokenizer_name)

    import sentencepiece as spm
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(f'{tokenizer_name}.model')
    txt_file_path = './data/tokenizer_test_medical.txt'
    res_csv_path = './data/eval_tokenizer.csv'
    eval_tokenizer(sp_model, txt_file_path,res_csv_path)