import torch
from torch import nn
import random

"""Implementation of abacus embeddings"""
# Example of how to extract digit tokens to pass into constructor
# digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])

class Abacus(torch.nn.Module):
    """
    Abacus Embeddings, learned emebddings resued for each digit.
    Integers must be reversed for this to work correctly.
    Transformers Can Do Arithmetic with the Right Embeddings, McLeish et al. (2024)
    """
    def __init__(self, tokenizer, embedding_dim, max_seq_length=1024, max_k=99):
        """
        digit_tokens (list): list of the tokens for each of the 10 digits, `digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])`
        embedding_dim (int): dimension to embed into
        max_seq_length (int): maximum number of embeddings that can be trained
        max_k (int): maximum k value which we randomly shift by during training
        """
        super().__init__()
        digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("digits", torch.tensor(digit_tokens), persistent=False)

        self.max_k = max_k

    def helper(self, mask, device):
        """
        Converts a binary mask of digit locations into spans of consecutive digits
        """
        mask_shape = mask.shape
        
        # Create a shifted version of the mask to detect changes from 0 to 1
        shifted_mask = torch.cat([torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype), mask[:, :-1]], dim=1)
        starts = (shifted_mask != mask) & mask  # 可以找到数字的起始位置
        
        # Generate IDs for each segment of 1s, processing row-wise
        segment_ids = torch.cumsum(starts, dim=1)  # 每行累加True
        
        # Generate an index array row-wise
        index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(device)  # 每一行建立[0,1,2,....]的位置索引
        
        # Reset index at the start of each segment
        reset_index = torch.zeros_like(mask).long()
        second_term = index * starts.long()   # 索引*mask
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)  # 把second_term的数字按segment_ids顺序排列，后面补0.
        
        # Calculate positions in segment
        positions = index - reset_index.gather(1, segment_ids) + 1
        
        # Ensure only values within 1-segments are non-zero
        result = positions * mask

        return result

    def forward(self, input_ids):
        """
        input_ids (tensor): a batch of inputs, each row is a sample
        """
        mask = torch.isin(input_ids, self.digits)
        output = self.helper(mask, input_ids.device)

        k=0
        if self.training:
            k = random.randint(0, self.max_k)
            output[output>0] += k # as we already have ones in the tensor, the tensor values will be k+1

        return self.embedding(output)
    


def get_embedding(embedding_type, vocab_size, hidden_dim, tokenizer=None):
    if embedding_type == 'default':
        return nn.Embedding(vocab_size, hidden_dim)
    elif embedding_type.lower() == 'abacus':  # 代码不全，不能用
        return Abacus(tokenizer, max_seq_length=vocab_size, embedding_dim=hidden_dim)
    else:
        return nn.Embedding(vocab_size, hidden_dim)
