import time

import torch
import torch.nn as nn

from src.quant.gptq.gptq import *
from src.share import init_model
from src.utils import find_layers
from src.quant.gptq.quant.quantizer import Quantizer
from src.quant.gptq.quant.quant_linear import QuantLinear
from src.quant.gptq.gptq import Observer
from src.quant.share import gptq_make_quant_linear
from src.quant.gptq.quant import make_quant_attn
from src.quant.gptq.quant import make_quant_norm
from src.quant.gptq.quant import make_fused_mlp
from src.quant.gptq.quant import autotune_warmup_linear
from src.quant.gptq.quant import autotune_warmup_fused

@torch.no_grad()
def quant_sequential(model, dataloader, dev):
    print('Starting ...')
    layers = model.model.layers

    model.model.tok_embeddings = model.model.tok_embeddings.to(dev) 
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.model.parameters())).dtype
    inps = torch.zeros(  # [128, 2048, 768]
        (args.nsamples, model.model.max_seq_len, model.model.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'position_ids': None, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp  # hidden_status
            cache['i'] += 1
            cache['position_ids'] = kwargs['position_ids'] # 
            cache['attention_mask'] = kwargs['attention_mask'] # mask
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:  # 循环处理，把inps和cache填满
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.tok_embeddings = model.model.tok_embeddings.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    position_ids = cache['position_ids']
    attention_mask = cache['attention_mask']

    print('Ready.')

    quantizers = {}
    observer = Observer()
    for i in range(len(layers)):  # 遍历每一个transformer layer
        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i].to(dev)

        subset = find_layers(layer)  # 找到所有子层(Linear层或Conv层)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])  # 每一个子层创建量化类
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name))) # 获取每一个子层的输入输出
        for j in range(args.nsamples): # 每组输入都处理一编，处理N组，得到N组的中间每一层的输入和输出
            outs[j] = layer(inps[j].unsqueeze(0), position_ids, attention_mask=attention_mask)[0] # 运行了gptq[name].add_batch()
        for h in handles:
            h.remove()

        for name in subset: # 对每一个子层进行量化
            scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
            quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

            if args.observe:
                observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
            else:
                gptq[name].free()

        for j in range(args.nsamples):  # 处理获取N组输出
            outs[j] = layer(inps[j].unsqueeze(0), position_ids, attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps  # 输入和输出反过来，下一个transformer继续用
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    if args.observe:
        observer.print()
        conditions = gen_conditions(args.wbits, args.groupsize)
        for item in observer.items():
            name = item[0]
            layerid = item[1]
            gptq = item[2]['gptq']
            error = item[2]['error']
            target = error / 2

            table = Texttable()
            table.header(['wbits', 'groupsize', 'error'])
            table.set_cols_dtype(['i', 'i', 'f'])
            table.add_row([args.wbits, args.groupsize, error])

            print('Optimizing {} {} ..'.format(name, layerid))
            for wbits, groupsize in conditions:

                if error < target:
                    # if error dropped 50%, skip
                    break

                gptq.quantizer.configure(wbits, perchannel=True, sym=args.sym, mse=False)

                scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name)

                table.add_row([wbits, groupsize, error])
                quantizers['model.layers.%d.%s' % (layerid, name)] = (gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            print(table.draw())
            print('\n')
            gptq.layer.to('cpu')
            gptq.free()

    return quantizers

@torch.no_grad()
def model_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.model.max_seq_len

    layers = model.model.layers
    model.model.tok_embeddings = model.model.tok_embeddings.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.model.max_seq_len, model.model.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'position_ids': None, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['position_ids'] = kwargs['position_ids']
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.model.max_seq_len):((i + 1) * model.model.max_seq_len)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.tok_embeddings = model.model.tok_embeddings.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    position_ids = cache['position_ids']
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), position_ids, attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.output = model.output.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        lm_logits = model.output(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.model.max_seq_len):((i + 1) * model.model.max_seq_len)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.model.max_seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.model.max_seq_len))
    print(ppl.item())


# TODO: perform packing on GPU
def model_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    gptq_make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)

    print('Done.')
    return model

def load_quant_model(checkpoint, wbits, groupsize, opt, fused_mlp=True, eval=True, warmup_autotune=False):
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    opt.load_in_lowbit = wbits
    opt.load_in_lowbit_groupsize = groupsize
    model, _ = init_model(opt)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['output']:
        if name in layers:
            del layers[name]
    gptq_make_quant_linear(model, layers, wbits, groupsize)

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    if eval:
        make_quant_attn(model)
        make_quant_norm(model)
        if fused_mlp:
            make_fused_mlp(model)

    if warmup_autotune:
        autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            autotune_warmup_fused(model)

    print('Done.')

    return model


def benchmark(model, input_ids, device, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else device)
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    with torch.no_grad():
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape((1,-1)),
                past_key_values=cache['past'],
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(device), input_ids[:, (i + 1)].to(device)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        import numpy as np
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())


if __name__ == '__main__':
    import argparse
    from src.data_prepare.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='llama model to load')
    parser.add_argument('--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--wbits', type=int, default=4, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--test-generation', action='store_true', help='test generation.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument('--observe',action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
                              When this feature enabled, `--save` or `--save_safetensors` would be disable.')
    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')


    args = parser.parse_args()
    from setting import get_parser_args, parser_model_config
    opt = get_parser_args(parser)
    opt, _ = parser_model_config(opt)
    
    if args.load:
        model = load_quant_model(args.load, args.wbits, args.groupsize, opt)
        args.model = args.load
    else:
        opt.model_path = opt.model
        opt.init_from = "resume"
        model, _ = init_model(opt)
        model.eval()

    from src.data_prepare.datautils import get_loaders
    print('load dataset......')
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, vocab_file=opt.vocab_file, seqlen=model.model.max_seq_len
    )

    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        print('quant_sequential......')
        quantizers = quant_sequential(model, dataloader, opt.device)
        print(time.time() - tick)

    if args.benchmark:
        model = model.to(opt.device)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, device=opt.device, check=args.check)

    if args.eval:
        datasets = ['wikitext2', 'ptb', 'c4']
        # datasets = ['wikitext2', 'ptb-new', 'c4-new']
        for dataset in datasets: 
            print(f'model_eval: {dataset}')
            dataloader, testloader = get_loaders(
                dataset, seed=0, model=args.model, seqlen=model.model.max_seq_len
            )
            print(dataset)
            model_eval(model, testloader, opt.device)

    if args.save:
        model_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), args.save) 

    if args.save_safetensors:
        model_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)