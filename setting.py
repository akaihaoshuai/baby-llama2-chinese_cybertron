from argparse import ArgumentParser
import yaml
import json

def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default='config/config.yaml', help="path to config")
    parser.add_argument("--ds_config", type=str, default='config/deepspeed.json', help="path to config")
    parser.add_argument("--save_path", type=str, default='', help="path to config")
    parser.add_argument("--train_data_path", type=list, default=['./data/pretrain_data.bin'], help="path to config")
    parser.add_argument("--valid_data_path", type=list, default=['./data/pretrain_data.bin'], help="path to config")
    parser.add_argument("--test_data_path", type=list, default=['./data/pretrain_data.bin'], help="path to config")
    parser.add_argument("--sft_data_path", type=str, default='./data/sft_data.csv', help="path to config")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--out_dir", type=str, default='out', help="path to config")
    parser.add_argument("--model_path", type=str, default='best.pth', help="path to config")
    
    # model param
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=0, help="0及其以下,则取n_heads的值,为MHQ.为1则是MQA,大于1且小于n_layers则为GQA")
    parser.add_argument("--use_bias", type=bool, default=False)
    parser.add_argument("--dtype", type=str, default='float16', help="path to config")
    parser.add_argument("--vocab_size", type=int, default=64793)
    parser.add_argument("--vocab_file", type=str, default='./chatglm_tokenizer/tokenizer.model', help="path to config")
    # train params
    parser.add_argument("--max_epoch", type=int, default=2)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument("--eval_only", type=bool, default=False)
    parser.add_argument("--always_save_checkpoint", type=bool, default=True)
    parser.add_argument("--init_from", type=str, default='scratch', help="path to config")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--multiple_of", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--decay_lr", type=bool, default=True)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--lr_decay_iters", type=int, default=80000)
    # learning rate decay settings
    parser.add_argument("--min_lr", type=float, default=1e-5)
    # DDP settings
    parser.add_argument("--backend", type=str, default='nccl', help="path to config")
    # system
    parser.add_argument("--device", type=str, default='cuda', help="path to config")
    parser.add_argument("--compile", type=bool, default=False)
    parser.add_argument("--local_rank", type=int, default=0)

    #eval
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shot", type=int, default=0, help='zero shot')

    parser.add_argument("--use_tensorboard", type=bool, default=True)

    opt = parser.parse_args()

    return opt

def parser_config(opt):
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    opt.save_path = config['save_path']
    opt.train_data_path = config['dataset_params']['train_data_path']
    opt.valid_data_path = config['dataset_params']['valid_data_path']
    opt.sft_data_path = config['dataset_params']['sft_data_path']
    opt.test_data_path = config['dataset_params']['test_data_path']
    opt.max_seq_len = config['dataset_params']['max_seq_len']
    
    opt.model_path = config['model_path']

    opt.dim = config['model_params']['dim']
    opt.n_layers = config['model_params']['n_layers']
    opt.n_heads = config['model_params']['n_heads']
    opt.n_kv_heads = config['model_params']['n_kv_heads']
    opt.use_bias = config['model_params']['use_bias']
    opt.dtype = config['model_params']['dtype']
    opt.vocab_size = config['model_params']['vocab_size']
    opt.vocab_file = config['model_params']['vocab_file']

    opt.max_epoch = config['train_params']['max_epoch']
    opt.eval_interval = config['train_params']['eval_interval']
    opt.log_interval = config['train_params']['log_interval']
    opt.eval_iters = config['train_params']['eval_iters']
    opt.eval_only = config['train_params']['eval_only']
    opt.always_save_checkpoint = config['train_params']['always_save_checkpoint']
    opt.init_from = config['train_params']['init_from']
    opt.grad_accum_steps = config['train_params']['grad_accum_steps']
    opt.batch_size = config['train_params']['batch_size']
    opt.multiple_of = config['train_params']['multiple_of']
    opt.dropout = config['train_params']['dropout']
    opt.learning_rate = config['train_params']['learning_rate']
    opt.weight_decay = config['train_params']['weight_decay']
    opt.beta1 = config['train_params']['beta1']
    opt.beta2 = config['train_params']['beta2']
    opt.grad_clip = config['train_params']['grad_clip']
    opt.decay_lr = config['train_params']['decay_lr']
    opt.warmup_iters = config['train_params']['warmup_iters']
    opt.lr_decay_iters = config['train_params']['lr_decay_iters']
    opt.min_lr = config['train_params']['min_lr']
    opt.backend = config['train_params']['backend']
    opt.device = config['train_params']['device']
    opt.compile = config['train_params']['compile']

    opt.max_new_tokens = config['eval_params']['max_new_tokens']
    opt.temperature = config['eval_params']['temperature']
    opt.top_k = config['eval_params']['top_k']
    opt.seed = config['eval_params']['seed']
    opt.shot = config['eval_params']['shot']

    return opt,config

def read_deepspeed_config(opt):
    with open(opt.ds_config) as f:
        ds_config = json.load(f)

    ds_config['optimizer']['params']['lr']=opt.learning_rate
    ds_config['optimizer']['params']['betas'][0]=opt.beta1
    ds_config['optimizer']['params']['betas'][0]=opt.beta2
    ds_config['optimizer']['params']['weight_decay']=opt.weight_decay

    # ds_config['train_batch_size']=opt.batch_size  # 如果设置了gradient_accumulation_steps和train_micro_batch_size_per_gpu，则忽略
    ds_config['train_micro_batch_size_per_gpu']=opt.batch_size  # 不考虑梯度处理
    ds_config['gradient_accumulation_steps']=opt.grad_accum_steps  # 梯度累积

    ds_config['scheduler']['params']['warmup_num_steps']=opt.warmup_iters  # 


    ds_config['steps_per_print']=opt.log_interval  # 
    ds_config['gradient_clipping']=opt.grad_clip  # 

    return ds_config