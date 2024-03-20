from src.quant.gptq.quant.quant_linear import QuantLinear

def gptq_make_quant_linear(module, names, bits, groupsize, name=''):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, QuantLinear(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
    for name1, child in module.named_children():
        gptq_make_quant_linear(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)


def bnb_make_quant_linear(module, names, bits, groupsize, name=''):
    import bitsandbytes as bnb
    from packaging import version
    import importlib.metadata
    supports_8bit = version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.39.0")
    supports_4bit = version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.41.3")

    if 8 == bits and supports_8bit:
       bnb_quant_linear = bnb.nn.Linear8bitLt
    elif 4 == bits and supports_4bit:
       bnb_quant_linear = bnb.nn.Linear4bit

    if isinstance(module, bnb_quant_linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, bnb_quant_linear(tmp.in_features, tmp.out_features, 
                                                      tmp.bias is not None,
                                                      has_fp16_weights=False, threshold=6.0))
    for name1, child in module.named_children():
        bnb_make_quant_linear(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)


def awq_make_quant_linear(module, names, bits, groupsize, name='', backend='autoawq'):
    from transformers.utils import AwqBackendPackingMethod
    if backend == AwqBackendPackingMethod.AUTOAWQ:
        from transformers.integrations.awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
        target_cls = WQLinear_GEMM if quantization_config.version == AWQLinearVersion.GEMM else WQLinear_GEMV
    else:
        from transformers.integrations.awq.quantize.qmodule import WQLinear
        target_cls = WQLinear

    if isinstance(module, target_cls):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, target_cls(w_bit=bits, group_size=groupsize, 
                                             in_features=tmp.in_features, 
                                             out_features=tmp.out_features, 
                                             bias=tmp.bias is not None))
    for name1, child in module.named_children():
        awq_make_quant_linear(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)


def bitnet_make_quant_linear(module, names, bits, groupsize, name=''):
    from src.quant.bitnet.bitbnet_b158 import BitLinearNew
    if isinstance(module, BitLinearNew):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, BitLinearNew(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
    for name1, child in module.named_children():
        bitnet_make_quant_linear(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)

