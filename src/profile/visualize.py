import matplotlib.pyplot as plt
from typing import List

def display_qk_heatmap_per_head(qk_per_token, prompt_split_as_tokens: List[str], name):
    for idx, qk_heatmaps in enumerate(qk_per_token):
        head_num = qk_heatmaps.shape[1]
        for head_idx in range(head_num):
            _, ax = plt.subplots()
            qk_heatmap = qk_heatmaps[0][head_idx]
            prompt_split_as_tokens = prompt_split_as_tokens[:qk_heatmap.shape[0]]
            im = ax.imshow(qk_heatmap, cmap='viridis')
            ax.set_xticks(range(len(prompt_split_as_tokens)))
            ax.set_yticks(range(len(prompt_split_as_tokens)))
            ax.set_xticklabels(prompt_split_as_tokens)
            ax.set_yticklabels(prompt_split_as_tokens)
            ax.figure.colorbar(im, ax=ax)
            plt.savefig(f'doc/heatmaps/{name}_layer{idx}_qkhead{head_idx}_heatmap.png')


def display_rope_freqs_cis(freqs_cis):
    value = freqs_cis[3]
    plt.figure()
    for i, element in enumerate(value[:17]):
        plt.plot([0, element.real], [0, element.imag], color='blue', linewidth=1, label=f"Index: {i}")
        plt.annotate(f"{i}", xy=(element.real, element.imag), color='red')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Plot of one row of freqs_cis')
    plt.show()