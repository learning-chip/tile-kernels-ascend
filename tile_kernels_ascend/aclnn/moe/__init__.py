import torch
from typing import Optional

try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    torch_npu = None
    NPU_AVAILABLE = False


def aux_fi(
    topk_idx: torch.Tensor,
    num_experts: int,
    num_aux_topk: int,
) -> torch.Tensor:
    num_tokens, num_topk = topk_idx.shape
    if num_tokens == 0:
        return torch.zeros(num_experts, dtype=torch.float32, device=topk_idx.device)
    valid_idx = topk_idx[topk_idx >= 0]
    counts = torch.bincount(valid_idx.view(-1), minlength=num_experts)
    return counts.float()[:num_experts] * num_experts / (num_tokens * num_aux_topk)


def group_count(
    group_idx: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    valid = group_idx[group_idx >= 0]
    return torch.bincount(valid.view(-1).to(torch.long), minlength=num_groups)[:num_groups].to(torch.int32)


def mask_indices_by_tp(
    indices: torch.Tensor,
    n: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
) -> torch.Tensor:
    per_gpu = n // num_ep_ranks
    per_dp = num_tp_ranks * per_gpu
    value = indices.clone()
    invalid = (value < 0) | ((value // per_gpu) % num_tp_ranks != tp_rank)
    value = value - tp_rank * per_gpu
    dp_rank = value // per_dp
    value = value - dp_rank * (per_dp - per_gpu)
    value[invalid | (value < 0)] = -1
    return value


def normalize_weight(
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    denom = topk_weights.sum(dim=-1).clamp(min=1e-20)
    return denom, topk_weights / denom.unsqueeze(1)


def inplace_unique_group_indices(
    group_indices: torch.Tensor,
    num_groups: int,
) -> None:
    num_tokens, num_topk = group_indices.shape
    vals, idx = torch.sort(group_indices, dim=1, stable=True)
    first = torch.ones((num_tokens, num_topk), dtype=torch.bool, device=group_indices.device)
    first[:, 1:] = vals[:, 1:] != vals[:, :-1]
    dup_sorted = ~first
    dup_orig = torch.zeros((num_tokens, num_topk), dtype=torch.bool, device=group_indices.device)
    dup_orig.scatter_(1, idx, dup_sorted)
    group_indices[dup_orig] = -1


def get_fused_mapping(
    num_expanded_tokens: int,
    num_tokens: int,
    num_topk: int,
    topk_idx: torch.Tensor,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    raise NotImplementedError("ACLNN backend: get_fused_mapping interface differs from torch_npu.npu_moe_init_routing API")


def topk_gate(
    logits: torch.Tensor,
    bias: torch.Tensor,
    num_topk: int,
    scoring_func: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scoring_func.lower() == 'softmax':
        result = torch_npu.npu_moe_gating_top_k_softmax(logits, k=num_topk)
    else:
        result = torch_npu.npu_moe_gating_top_k(logits, k=num_topk)
    # NPU returns (weights, expert_idx, row_idx); torch ref returns (topk_idx, topk_weights)
    weights, expert_idx, _row_idx = result
    return expert_idx.long(), weights


def top2_sum_gate(
    logits: torch.Tensor,
    bias: torch.Tensor,
    num_topk: int,
    num_topk_groups: int,
    num_groups: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
    scoring_func: str,
    mask: Optional[torch.Tensor] = None,
    fix_routing_mask: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
    unmapped_topk_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_groups > 0 and num_topk_groups < num_groups:
        result = torch_npu.npu_moe_gating_top_k(
            logits,
            k=num_topk,
            k_group=num_topk_groups,
            group_count=num_groups,
            group_select_mode=1,
            routed_scaling_factor=routed_scaling_factor,
        )
    else:
        result = torch_npu.npu_moe_gating_top_k(
            logits,
            k=num_topk,
            routed_scaling_factor=routed_scaling_factor,
        )
    # NPU returns (weights, expert_idx, row_idx); torch ref returns (topk_idx, topk_weights)
    weights, expert_idx, _row_idx = result
    return expert_idx.long(), weights


def topk_sum_and_topk_group_idx(
    scores: torch.Tensor,
    num_group_sum_topk: int,
    num_topk_groups: int,
) -> torch.Tensor:
    raise NotImplementedError("ACLNN backend: topk_sum_and_topk_group_idx not directly available (use npu_moe_gating_top_k for full pipeline)")


def expand_to_fused(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("ACLNN backend: expand_to_fused interface differs from torch_npu.npu_moe_init_routing (different input/output layout)")


def expand_to_fused_with_sf(
    x: tuple[torch.Tensor, torch.Tensor],
    num_per_channels: int,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
    use_tma_aligned_col_major_sf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("ACLNN backend: expand_to_fused_with_sf not yet implemented")


def reduce_fused(
    x: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    token_topk_to_pos: torch.Tensor,
    fp8_format: str = '',
    sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    raise NotImplementedError("ACLNN backend: reduce_fused interface differs from torch_npu.npu_moe_finalize_routing (different argument layout)")
