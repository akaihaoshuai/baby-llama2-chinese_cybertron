from src.utils import *
from src.models.model_args import ModelArgs,LoraArgs
from src.models import _get_model_architecture, _get_tokenizer


def init_model(model_config=None, model_path=None, tokenizer=None, lora_config=None, flag='train'):
    if model_path is None:
        # init a new model from scratch
        print_rank_0("Initializing a new model from scratch")
        model_args = ModelArgs(**model_config) if model_config is not None else ModelArgs()
        if tokenizer is None:
            tokenizer = _get_tokenizer(model_args.architecture, None)
        model_args.bos_id = tokenizer.tokenizer.bos_id
        model_args.eos_id = tokenizer.tokenizer.eos_id
        model_args.pad_id = tokenizer.tokenizer.pad_id
        model_architecture = _get_model_architecture(model_args.architecture)
        model = model_architecture(model_args, flag)
    else:
        # resume training from a checkpoint.
        model_path_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        
        max_epoch = -1
        epoch_bin_path_list = [sub_path for sub_path in os.listdir(model_path_dir) if 'epoch' in sub_path]
        if len(epoch_bin_path_list) == 0:
            epoch_bin_path_list = [sub_path for sub_path in os.listdir(model_path_dir) if 'iter' in sub_path]
            for bin_path in epoch_bin_path_list:
                epoch_num = int(bin_path.split('_')[-1].split('.')[0])
                if max_epoch < epoch_num:
                    max_epoch = epoch_num
            ckpt_path = os.path.join(model_path, f'iter_{max_epoch}.pth')
        else:
            for bin_path in epoch_bin_path_list:
                epoch_num = int(bin_path.split('_')[-1].split('.')[0])
                if max_epoch < epoch_num:
                    max_epoch = epoch_num
            ckpt_path = os.path.join(model_path, f'epoch_{max_epoch}.pth')


        print_rank_0(f'load model from path: {ckpt_path}.')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'model_config' in checkpoint:
            checkpoint_model_config = checkpoint["model_config"]
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ["hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
                model_config[k] = checkpoint_model_config[k]

        # create the model
        model_args = ModelArgs(**model_config) if model_config is not None else ModelArgs()
        lora_args = LoraArgs(**lora_config) if lora_config is not None else None
        if tokenizer is None:
            tokenizer = _get_tokenizer(model_args.architecture, model_path)
            if tokenizer is None:
                print_rank_0(f'model_path: {model_path} is None.')
                return None

        model_args.bos_id = tokenizer.tokenizer.bos_id
        model_args.eos_id = tokenizer.tokenizer.eos_id
        model_args.pad_id = tokenizer.tokenizer.pad_id
        model_architecture = _get_model_architecture(model_args.architecture)
        model = model_architecture(model_args, lora_args, flag)

        if 'model' in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model_dict = model.state_dict()
        for name, param in state_dict.items():
            if name in model_dict:
                if model_dict[name].shape == param.shape:
                    model_dict[name].copy_(param)
                elif model_dict[name].shape == param.T.shape:
                    model_dict[name].copy_(param.T)
                else:
                    print('load_weight shape error.')
                    return

    return model, tokenizer


def set_model_train(model):
    if model.lora_args is None:
        model.train()
    else:
        # 冻结非lora模块的梯度
        for name, param in model.named_parameters():
            if 'lora' in name or 'dora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


def set_model_eval(model):
    model.eval()



def eval_model(model, ctx=None):
    from src.chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
    tokenizer=ChatGLMTokenizer()

    model.eval()
    device = next(model.parameters()).device

    data = [
        {"question": "最近我在办公室坐久了会感到头晕，请问这是什么原因?有什么缓解办法吗？", "target": ""},
        # {"question": "前列腺囊肿的症状是什么？", "target": ""},
        # {"question": "请问，世界上最大的动物是什么？", "target": ""},
    ]
    if ctx is None:
         ctx = get_ctx(device)

    for p in data:
        # run generation
        prompt=p['question']
        x=tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.special_tokens['<bos>']]
        x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
        target = p['target']
        with torch.no_grad():
            with ctx:
                y = model.generate(x)
                answer=tokenizer.decode(y[0])
                answer=answer.replace(prompt,'')
                print_rank_0(f'[prompt]: {prompt}')
                print_rank_0(f'[answer]: {answer}')
                print_rank_0('---------------')