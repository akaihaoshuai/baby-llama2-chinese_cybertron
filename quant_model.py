import time

import torch
import torch.nn as nn

from src.gptq.gptq import *
from src.gptq.quant import *
from src.share import init_model
from src.utils import find_layers


def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def quant_sequential(model, dataloader, dev):
    print('Starting ...')
    layers = model.model.layers

    model.model.tok_embeddings = model.model.tok_embeddings.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.model.parameters())).dtype
    inps = torch.zeros(  # [128, 2048, 768]
        (args.nsamples, model.model.max_seq_len, model.model.dim), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp  # hidden_status
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask'] # mask
            cache['position_ids'] = kwargs['position_ids'] # mask
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
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):  # 遍历每一个transformer layer
        layer = layers[i].to(dev)

        subset = find_layers(layer)  # 找到所有子层(Linear层或Conv层)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])  # 每一个子层创建量化类
            gptq[name].quantizer = Quantizer()
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
            print(i, name)
            print('Quantizing ...')
            gptq[name].fasterquant(
                percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
            )
            quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free()
        for j in range(args.nsamples):  # 处理获取N组输出
            outs[j] = layer(inps[j].unsqueeze(0), position_ids, attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps  # 输入和输出反过来，下一个transformer继续用

    return quantizers

@torch.no_grad()
def model_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    layers = model.model.decoder.layers

    model.model.tok_embeddings = model.model.tok_embeddings.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.tok_embeddings = model.model.decoder.tok_embeddings.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
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
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())


# TODO: perform packing on GPU
def model_pack(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, faster=args.faster_kernel)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)

    print('Done.')
    return model

def load_quant_model(model, checkpoint, opt):
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    model = init_model(opt)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['output']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, faster=args.faster_kernel)

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))
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
    for i, layer in enumerate(model.model.decoder.layers):
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
        attention_mask = torch.ones((1, input_ids.numel()), device=device)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape((1,-1)),
                past_key_values=cache['past'],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
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
    from src.data.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='quant.ckpt',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    args = parser.parse_args()
    from setting import get_parser_args, parser_model_config
    opt = get_parser_args(parser)
    opt, _ = parser_model_config(opt)
    
    if args.load:
        model = load_quant_model(args.model, args.load, opt)
    else:
        model = init_model(opt)
        model.eval()

    from src.data.datautils import get_loaders
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, model=args.model, seqlen=model.model.max_seq_len
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = quant_sequential(model, dataloader, opt.device)
        print(time.time() - tick)

    if args.benchmark:
        model = model.to(opt.device)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, device=opt.device, check=args.check)
    if args.load:
        exit()

    datasets = ['wikitext2', 'ptb', 'c4']
    if args.new_eval:
      datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, seed=0, model=args.model, seqlen=model.model.max_seq_len
        )
        print(dataset)
        model_eval(model, testloader, opt.device)

    if args.save:
        model_pack(model, quantizers)
        torch.save(model.state_dict(), args.save) 
