import torch
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

FONTSIZE = 16

font_config = {'font.size': FONTSIZE, 'font.family': 'DejaVu Math TeX Gyre'}
plt.rcParams.update(font_config)
plt.rcParams["figure.figsize"] = (4, 4.5)


# 模型名称，可以根据需要选择不同的模型尺寸
model_path = "/Meta-Llama-3-8B"
model_name = model_path.split('/')[-1]
# 加载模型和分词器
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model = model.eval()

# 要进行推理的文本
text = "Passage 1: Betty Hall Beatrice Perin Barker Hall (March 18, 1921 - April 26, 2018) was an American politician from the state of New Hampshire. Hall served in the New Hampshire House of Representatives for a total of 28 years, serving non-consecutively from 1970 until 2008. Hall grew up in New York City, where she attended Barnard College. In 1948, she and her husband moved to the town of Brookline, New Hampshire, where they started a textile manufacturing firm. Beginning in the 1950s and early 1960s, Hall began participating in local politics, serving on several boards and commissions in Brookline. Hall was elected to the Brookline school board in 1963, and in 1972, she was elected to the town board of selectmen. Hall's career in statewide politics began in 1970 when she was elected to the New Hampshire House of Representatives as a member of the Republican Party. In 1986, Hall switched her party affiliation to the Democratic Party, citing the Republican Party's shift towards conservatism during the Reagan Era. During her political career, Hall was described as a firebrand who frequently bucked her party. While a Republican, Hall was seen as a liberal member of that party, and was considered to be a political enemy by conservative leaders. In the Democratic Party, Hall was a member of the grassroots base, challenging the party's establishment in a 2007 campaign for chairman of the New Hampshire Democratic Party."
# 编码文本
input_ids = tokenizer.encode(text, return_tensors="pt")

# generate kv cache and attention
outputs = model(input_ids, use_cache=True, output_attentions=True)
past_key_values = outputs.past_key_values
# attentions = outputs.attentions
torch.save(past_key_values, f'./kvcache_vis_{model_name}_kvcache.pt')
# torch.save(attentions, f'./kvcache_vis_{model_name}_attention.pt')

kv_filename = f'./kvcache_vis_{model_name}_kvcache.pt'
# attn_filename = f'./kvcache_vis_{model_name}_attention.pt'
kvcache = torch.load(kv_filename, map_location='cpu')
# attentions = torch.load(attn_filename, map_location='cpu')


save_path = 'kvcache_vis_figs'
os.makedirs(save_path, exist_ok=True)

for layer_id in range(len(kvcache)): # replace with your layer ids
    head_id = 0
    k, v = kvcache[layer_id][0].squeeze(0), kvcache[layer_id][1].squeeze(0)

    k = k.transpose(0, 1).abs().detach().numpy()
    v = v.transpose(0, 1).abs().detach().numpy()
    k, v = k[:, head_id, :], v[:, head_id, :]

    # Sample 2D tensor (replace this with your actual tensor)
    for idx, tensor in enumerate([k, v]):
        # Creating a meshgrid
        tokens, channels = tensor.shape
        x = np.arange(tokens)
        y = np.arange(channels)
        X, Y = np.meshgrid(x, y)
        # Creating a figure and a 3D subplot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plotting the surface
        surf = ax.plot_surface(X, Y, tensor.T, cmap='coolwarm')

        ax.xaxis.set_tick_params(pad=-5)
        ax.yaxis.set_tick_params(pad=-3)
        ax.zaxis.set_tick_params(pad=-130)

        # Adding labels
        ax.set_xlabel('Token', labelpad=-5)
        ax.set_ylabel('Column', labelpad=-1)
        if layer_id in [3, 16]:
            ax.zaxis.set_rotate_label(False) 
        if idx == 0:
            save_filename = f'./{save_path}/{model_name}_layer{layer_id}_head{head_id}_k.pdf'
        else:
            save_filename = f'./{save_path}/{model_name}_layer{layer_id}_head{head_id}_v.pdf'
        plt.savefig(save_filename, bbox_inches='tight')
        plt.clf()