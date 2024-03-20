import json
import glob
import torch
import torch.nn as nn
import pandas as pd
import os

def check_is_processed(target_name, data_path):
    data_path_list = os.listdir(data_path)
    for data_name in data_path_list:
        if target_name.split('/')[-1] in data_name:
            return True

    return False

def getAllFiles(data_path):
    all_data_path_list = []
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            all_data_path_list.append(file_path)
    return all_data_path_list

def read_ckpt(model_path_name, lora_path=''):
    state_dict = None
    lora_state_dict = None
    
    if lora_path=='':
        if model_path_name.endswith('.pth'):
            model_path = model_path_name
        elif model_path_name.endswith('.lora'):
            lora_path = model_path_name

            lora_name = model_path_name.split('/')[-1]
            pretrain_model_path = model_path_name.replace('lora_ft', 'pretrain')
            name_char_list = lora_name.split('_')
            pretrain_model_name = name_char_list[1]+'_'+name_char_list[2]+'.pth'
            pretrain_model_path_name = pretrain_model_path.replace(lora_name, pretrain_model_name)
            if os.path.exists(pretrain_model_path_name):
                model_path = pretrain_model_path_name
            else:
                model_path = ''
    else:
        model_path = model_path_name

    if model_path != '':
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if lora_path != '':
        lora_state_dict = torch.load(lora_path, map_location=torch.device('cpu'))

    return model_path, state_dict, lora_path, lora_state_dict
    
def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def load_weight(model, state_dict, lora_state_dict=None, merge_lora_on_load=False, strict=True):
    # 从预训练权重中更新模型参数
    if model.model.params.ft_type == 'lora' or model.model.params.lora_path != '':
        model_dict = model.state_dict()
        for name, param in state_dict.items():
            if model_dict[name].shape == param.shape:
                model_dict[name].copy_(param)
            elif model_dict[name].shape == param.T.shape:
                model_dict[name].copy_(param.T)
            else:
                print('load_weight shape error.')
                return

        if lora_state_dict is not None:
            if merge_lora_on_load:
                from src.loralib.utils import merge_lora_on_load_func
                merge_lora_on_load_func(model, lora_state_dict)
            else:
                for name, param in lora_state_dict.items():
                    model_dict[name].copy_(param)
    else:
        model.load_state_dict(state_dict, strict)


def save_model(model, path, save_type='all', merge_lora=False):
    if merge_lora:
        from src.loralib.utils import merge_lora_to_save_func
        merge_lora_to_save_func(model, path)
        torch.save(model.state_dict(), path)
    else:
        if save_type == 'all':
            model_state_dict = model.state_dict()
            state_dict = {k: model_state_dict[k] for k in model_state_dict if model_state_dict[k] is not None}
            torch.save(state_dict, path)
        elif save_type == 'lora':
            model_state_dict = model.state_dict()
            lora_only = {k: model_state_dict[k] for k in model_state_dict if 'lora_' in k and model_state_dict[k] is not None}
            torch.save(lora_only, path.replace('.pth', '.lora'))
        else:
            model_state_dict = model.state_dict()
            state_dict = {k: model_state_dict[k] for k in model_state_dict if model_state_dict[k] is not None}
            torch.save(state_dict, path)
