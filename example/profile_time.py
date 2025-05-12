import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from src.utils import *
from src.model_runner import init_model

# 绘制折线图
def draw_plot(x, y, png_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, color='blue', label='line1')
    # ax.plot(x, y2, color='red', label='line2')
    ax.set_title('run time')
    ax.set_xlabel('batch_size')
    ax.set_ylabel('seconds')
    ax.legend()
    plt.savefig(f'{png_path}.png')
    plt.show()


def profile_time(attention, batch_size, seq_len, hidden_size, number):
    x = torch.randn(batch_size, seq_len, hidden_size)

    # 预热
    output = attention(x)

    t1 = benchmark.Timer(
        stmt="attention(x)",
        setup="x = torch.randn(batch_size, seq_len, hidden_size)",
        globals={"attention": attention, "batch_size": batch_size,
                  "seq_len": seq_len, "hidden_size": hidden_size},
    )

    mean_time = t1.timeit(number=number).mean
    return mean_time


# 使用示例
model,  _ = init_model()
attention = model.layers[0].attention

seq_len = model.params.max_seq_len
hidden_size = model.params.dim
number = 20

para_num = sum(p.numel() for p in attention.parameters())
print_rank_0(f"AttentionModule: {para_num/1048576:.3f} M.")

# 使用 torch.utils.benchmark 分析性能
with torch.no_grad():
    # 测量batch_size性能
    batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128]
    time_list = []
    for batch_size in batch_size_list:
        print_rank_0(f'batch_size: {batch_size}.')
        mean_time = profile_time(attention, batch_size, seq_len, hidden_size, number)
        time_list.append(mean_time)
        print_rank_0(f"Attention layer execution time: {mean_time:.6f} seconds for {number} times")
    draw_plot(batch_size_list, time_list, 'doc/profile/bs_time_profile')


    # 测量seq_len性能
    seq_len_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    batch_size = 4
    time_list = []
    for seq_len in seq_len_list:
        print_rank_0(f'seq_len: {seq_len}.')
        mean_time = profile_time(attention, batch_size, seq_len, hidden_size, number)
        time_list.append(mean_time)
        print_rank_0(f"Attention layer execution time: {mean_time:.6f} seconds for {number} times")
    draw_plot(seq_len_list, time_list, 'doc/profile/seqlen_time_profile')