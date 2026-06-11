import torch
from typing import Optional, Union
from tile_kernels_ascend.torch.utils import align
from tile_kernels_ascend.torch.quant.types import QuantTensor


def expand_to_fused(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden = x.shape
    num_expanded_tokens = pos_to_expert.shape[0]
    out = torch.zeros((num_expanded_tokens, hidden), dtype=x.dtype, device=x.device)
    pos_flat = token_topk_to_pos.reshape(-1)
    mask = pos_flat >= 0
    valid_pos = pos_flat[mask]
    num_topk = token_topk_to_pos.shape[1]
    x_repeated = x.unsqueeze(1).expand(-1, num_topk, -1).reshape(-1, hidden)
    out[valid_pos] = x_repeated[mask]
    return out


def expand_to_fused_with_sf(
    x: QuantTensor,
    num_per_channels: int,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
    use_tma_aligned_col_major_sf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_data, x_sf = x
    num_tokens, hidden = x_data.shape
    num_expanded_tokens = pos_to_expert.shape[0]
    hidden_sf = x_sf.shape[1]
    out = torch.zeros((num_expanded_tokens, hidden), dtype=x_data.dtype, device=x_data.device)
    if use_tma_aligned_col_major_sf:
        num_expanded_sf_tokens = align(num_expanded_tokens, 4)
        out_sf = torch.zeros((hidden_sf, num_expanded_sf_tokens), dtype=x_sf.dtype, device=x_sf.device)
        out_sf = out_sf[:, :num_expanded_tokens]
        out_sf = out_sf.T
    else:
        out_sf = torch.zeros((num_expanded_tokens, hidden_sf), dtype=x_sf.dtype, device=x_sf.device)
    num_topk = token_topk_to_pos.shape[1]
    pos_flat = token_topk_to_pos.reshape(-1)
    mask = pos_flat >= 0
    valid_pos = pos_flat[mask]
    x_data_rep = x_data.unsqueeze(1).expand(-1, num_topk, -1).reshape(-1, hidden)
    out[valid_pos] = x_data_rep[mask]
    x_sf_rep = x_sf.unsqueeze(1).expand(-1, num_topk, -1).reshape(-1, hidden_sf)
    out_sf[valid_pos] = x_sf_rep[mask]
    return out, out_sf
