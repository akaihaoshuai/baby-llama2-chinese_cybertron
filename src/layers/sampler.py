import torch
from torch import nn

class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence and frequency penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self) -> None:
        super().__init__()

    def _apply_top_p_top_k(
        self,
        logits: torch.Tensor,
        top_p: float=0.4, 
        top_k: int=10, 
    ) -> torch.Tensor:
        logits_sort, logits_idx = logits.sort(dim=-1, descending=True)
        if top_p < 1.0:
            p = torch.tensor(top_p, dtype=torch.float, device=logits.device)

            # Apply top-p.
            probs_sort = logits_sort.softmax(dim=-1)
            probs_sum = probs_sort.cumsum(dim=-1)
            top_p_mask = (probs_sum - probs_sort) > p
            logits_sort[top_p_mask] = -float("inf")
        
        # Apply top-k.
        if top_k > 0:
            k = torch.tensor(top_k, dtype=torch.int, device=logits.device)
            # Create a mask for the top-k elements.
            top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
            top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
            top_k_mask = top_k_mask >= k
            logits_sort[top_k_mask] = -float("inf")

        # Re-sort the probabilities.
        logits = torch.gather(logits_sort,
                              dim=-1,
                              index=torch.argsort(logits_idx, dim=-1))
        return logits

    def forward(
        self,
        logits: torch.Tensor,
        temperature: float=1.0, 
        top_p: float=0.4, 
        top_k: int=10, 
    ):
        if temperature != 1.0:
            t = torch.tensor(temperature,
                             dtype=logits.dtype,
                             device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        logits = self._apply_top_p_top_k(logits, top_p, top_k)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        pred_token_idx = probs.argmax(dim=-1).unsqueeze(1)

        return pred_token_idx