import torch
from typing import Dict, Sequence
import transformers
import copy

PROMPT_FIELD = 'prompt'
OUTPUT_FIELD = 'output'
IGNORE_INDEX = -100
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    # TODO: batch encode
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            #padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    mode: str
) -> Dict:
    """Preprocess the data by tokenizing."""
    samples = [s + t for s, t in zip(sources, targets)]
    samples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (samples, sources)]
    input_ids = samples_tokenized["input_ids"]
    # FIXME: sentencepiece case
    if mode == "sft":
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
    elif mode == "pretrain":
        labels = copy.deepcopy(input_ids)
    else:
        raise ValueError('Unvalid training mode.')

    # shift
    return dict(
        input_ids=[ids[: -1] for ids in input_ids],
        labels=[lbs[1: ]for lbs in labels]
    )

class DataCollatorForDataset(object):
    """Collate for supervised fine-tuning."""

    mode: str

    def get_attn_mask(self, input_ids):
        """
        Get triangular attention mask for a given sequence length / device.
        """
        bs = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        # lower triangular attention mask
        mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
            bs, 1, seq_length, seq_length
        )
        # convert to binary
        return mask < 0.5

    def get_position_ids(self, input_ids):
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [sample[PROMPT_FIELD] for sample in samples]
        targets = [sample[OUTPUT_FIELD] for sample in samples]

        data_dict = preprocess(sources, targets, self.tokenizer, self.mode)
        input_ids = data_dict["input_ids"]
        labels = data_dict["labels"]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        labels = torch.where(labels == self.tokenizer.pad_token_id, IGNORE_INDEX, labels)

        return (
            (
                input_ids,
                self.get_position_ids(input_ids),
                self.get_attn_mask(input_ids),
            ),
            labels
        )