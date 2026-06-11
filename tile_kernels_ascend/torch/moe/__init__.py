import torch
import torch.nn.functional as F
from enum import IntEnum
from typing import Optional


class ScoringFunc(IntEnum):
    SIGMOID = 0
    SQRTSOFTPLUS = 1
    SOFTMAX = 2
    IDENTITY = 3

    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, label: str):
        try:
            return cls[label.upper()]
        except KeyError:
            raise ValueError(f'{label} is not a valid {cls.__name__}')


def stable_topk(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    _, sorted_indices = torch.sort(scores, dim=1, descending=True, stable=True)
    return sorted_indices[:, :num_topk].contiguous()


def topk_sum_and_topk_group_idx(
    scores: torch.Tensor,
    num_group_sum_topk: int,
    num_topk_groups: int,
) -> torch.Tensor:
    group_scores_ref = scores.topk(num_group_sum_topk, dim=-1, sorted=False).values.sum(-1)
    return stable_topk(group_scores_ref, num_topk_groups)


def topk_gate_ref(
    logits: torch.Tensor,
    bias: torch.Tensor,
    num_topk: int,
    scoring_func: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    scoring = ScoringFunc.from_str(scoring_func)
    if scoring == ScoringFunc.SIGMOID:
        scores = torch.sigmoid(logits)
    elif scoring == ScoringFunc.SOFTMAX:
        scores = torch.softmax(logits, dim=-1)
    elif scoring == ScoringFunc.SQRTSOFTPLUS:
        scores = F.softplus(logits).sqrt()
    else:
        raise ValueError(f'Unknown scoring function: {scoring_func}')
    scores_biased = scores + bias.unsqueeze(0)
    topk_idx = stable_topk(scores_biased, num_topk)
    topk_weights = scores.gather(1, topk_idx)
    return topk_idx, topk_weights


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
    num_tokens_full, num_routed_experts = logits.shape
    scoring = ScoringFunc.from_str(scoring_func)
    if not use_shared_as_routed:
        num_shared_experts = 0
    num_physical_topk = num_topk + num_shared_experts
    num_logical_experts = num_routed_experts + num_shared_experts
    device = logits.device
    topk_idx_out = torch.full((num_tokens_full, num_physical_topk), -1, dtype=torch.int64, device=device)
    topk_weights_out = torch.zeros((num_tokens_full, num_physical_topk), dtype=torch.float32, device=device)
    if num_tokens_full == 0:
        return topk_idx_out, topk_weights_out
    active = mask if mask is not None else torch.ones(num_tokens_full, dtype=torch.bool, device=device)
    active_indices = active.nonzero(as_tuple=False).squeeze(1)
    num_tokens = active_indices.numel()
    if num_tokens == 0:
        if unmapped_topk_idx is not None:
            unmapped_topk_idx[~active] = -1
        return topk_idx_out, topk_weights_out
    logits_a = logits[active_indices]
    bias_b = bias.unsqueeze(0)
    if scoring == ScoringFunc.SIGMOID:
        scores_wo_bias = torch.sigmoid(logits_a)
    elif scoring == ScoringFunc.SQRTSOFTPLUS:
        scores_wo_bias = F.softplus(logits_a).sqrt()
    else:
        scores_wo_bias = torch.softmax(logits_a, dim=-1)
    scores_biased = (logits_a + bias_b) if scoring == ScoringFunc.SOFTMAX else (scores_wo_bias + bias_b)
    fix_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    if fix_routing_mask is not None and unmapped_topk_idx is not None:
        fix_mask = fix_routing_mask[active_indices]
    topk_idx_local = torch.full((num_tokens, num_topk), -1, dtype=torch.int64, device=device)
    topk_score_local = torch.zeros((num_tokens, num_topk), dtype=torch.float32, device=device)
    normal_mask = ~fix_mask
    if normal_mask.any():
        normal_indices = normal_mask.nonzero(as_tuple=False).squeeze(1)
        sb = scores_biased[normal_indices]
        if num_groups != num_topk_groups:
            num_per_group = num_routed_experts // num_groups
            top_group_idx = topk_sum_and_topk_group_idx(sb.view(-1, num_groups, num_per_group), 2, num_topk_groups)
            group_mask = torch.ones((normal_indices.numel(), num_groups), dtype=torch.bool, device=device)
            group_mask.scatter_(1, top_group_idx, False)
            sb = sb.masked_fill(
                group_mask.unsqueeze(-1).expand(-1, num_groups, num_per_group).reshape(-1, num_routed_experts),
                float('-inf'),
            )
        selected = stable_topk(sb, num_topk)
        topk_idx_local[normal_indices] = selected
        topk_score_local[normal_indices] = scores_wo_bias[normal_indices].gather(1, selected)
    if fix_mask.any() and unmapped_topk_idx is not None:
        fix_indices = fix_mask.nonzero(as_tuple=False).squeeze(1)
        pre_idx = unmapped_topk_idx[active_indices[fix_indices]]
        topk_idx_local[fix_indices] = pre_idx
        topk_score_local[fix_indices] = scores_wo_bias[fix_indices].gather(1, pre_idx.clamp(min=0))
    if unmapped_topk_idx is not None:
        unmapped_topk_idx[active_indices] = topk_idx_local
        if mask is not None:
            unmapped_topk_idx[~active] = -1
    topk_sum = topk_score_local.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    topk_weights_routed = topk_score_local / topk_sum * routed_scaling_factor
    if num_shared_experts > 0:
        shared_idx = torch.arange(num_routed_experts, num_logical_experts, dtype=torch.int64, device=device)
        topk_idx_all = torch.cat([topk_idx_local, shared_idx.expand(num_tokens, -1)], dim=1)
        topk_weights_all = torch.cat([
            topk_weights_routed,
            torch.ones((num_tokens, num_shared_experts), dtype=torch.float32, device=device),
        ], dim=1)
    else:
        topk_idx_all, topk_weights_all = topk_idx_local, topk_weights_routed
    if to_physical_map is not None and logical_count is not None:
        for lane in range(num_physical_topk):
            logical = topk_idx_all[:, lane]
            valid = logical >= 0
            if valid.any():
                global_idx = active_indices[valid].to(torch.int64)
                dup_idx = (ep_rank + global_idx * 23333) % logical_count[logical[valid]].to(torch.int64)
                topk_idx_all[valid, lane] = to_physical_map[logical[valid], dup_idx].to(torch.int64)
    num_extra = to_physical_map.shape[1] - 1 if to_physical_map is not None else 0
    experts_per_rank = (num_routed_experts + num_extra) // num_ep_ranks
    experts_per_dp = experts_per_rank * num_tp_ranks
    idx = topk_idx_all
    valid = idx >= 0
    ep_of = torch.where(valid, idx // experts_per_rank, torch.zeros_like(idx))
    idx = torch.where(valid & (ep_of % num_tp_ranks != tp_rank), -1, idx)
    valid = idx >= 0
    local = idx - tp_rank * experts_per_rank
    dp_of = torch.where(valid, local // experts_per_dp, torch.zeros_like(local))
    remapped = local - dp_of * (experts_per_dp - experts_per_rank)
    idx = torch.where(valid & (remapped >= 0), remapped, torch.where(valid, -1, idx))
    topk_idx_out[active_indices] = idx
    topk_weights_out[active_indices] = topk_weights_all
    return topk_idx_out, topk_weights_out


def aux_fi(topk_idx: torch.Tensor, num_experts: int, num_aux_topk: int) -> torch.Tensor:
    num_tokens, num_topk = topk_idx.shape
    if num_tokens == 0:
        return torch.zeros(num_experts, dtype=torch.float32, device=topk_idx.device)
    valid_idx = topk_idx[topk_idx >= 0]
    counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_idx.device)
    counts.scatter_add_(0, valid_idx, torch.ones_like(valid_idx))
    return counts.float() * num_experts / (num_tokens * num_aux_topk)


def group_count(group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    valid_idx = group_idx[group_idx >= 0]
    counts = torch.zeros(num_groups, dtype=torch.int32, device=group_idx.device)
    counts.scatter_add_(0, valid_idx, torch.ones_like(valid_idx, dtype=torch.int32))
    return counts


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


def normalize_weight(topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_topk = topk_weights.shape
    denominator = torch.full((num_tokens,), 1e-20, dtype=torch.float32, device=topk_weights.device)
    for k in range(num_topk):
        denominator = denominator + topk_weights[:, k]
    normalized_weights = topk_weights / denominator.unsqueeze(1)
    return denominator, normalized_weights


def inplace_unique_group_indices(group_indices: torch.Tensor, num_groups: int) -> None:
    num_tokens, num_topk = group_indices.shape
    vals, idx = torch.sort(group_indices, dim=1, stable=True)
    first_in_sorted = torch.ones((num_tokens, num_topk), dtype=torch.bool, device=group_indices.device)
    first_in_sorted[:, 1:] = vals[:, 1:] != vals[:, :-1]
    dup_in_sorted = ~first_in_sorted
    dup_in_orig = torch.zeros((num_tokens, num_topk), dtype=torch.bool, device=group_indices.device)
    dup_in_orig.scatter_(1, idx, dup_in_sorted)
    group_indices[dup_in_orig] = -1


def get_fused_mapping(
    num_expanded_tokens: int,
    num_tokens: int,
    num_topk: int,
    topk_idx: torch.Tensor,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    num_experts = topk_idx.max().item() + 1 if topk_idx.numel() > 0 else 0
    counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_idx.device)
    valid = topk_idx >= 0
    for k in range(num_topk):
        mask = valid[:, k]
        if mask.any():
            idx_k = topk_idx[mask, k]
            ones = torch.ones(mask.sum(), dtype=torch.int32, device=topk_idx.device)
            counts.scatter_add_(0, idx_k, ones)
    pos_to_expert = torch.full((num_expanded_tokens,), -1, dtype=torch.int32, device=topk_idx.device)
    token_topk_to_pos = torch.full((num_tokens, num_topk), -1, dtype=torch.int32, device=topk_idx.device)
    expert_offsets = torch.zeros(num_experts, dtype=torch.int32, device=topk_idx.device)
    offset = 0
    for e in range(num_experts):
        expert_offsets[e] = offset
        offset += counts[e]
    actual_expanded = offset
    aligned_expanded = ((offset + alignment - 1) // alignment) * alignment
    expert_counts = counts.cpu().numpy()
    expert_offsets_cpu = expert_offsets.cpu().numpy()
    pos_to_expert_cpu = torch.full((aligned_expanded,), -1, dtype=torch.int32, device='cpu').numpy()
    token_topk_to_pos_cpu = torch.full((num_tokens, num_topk), -1, dtype=torch.int32, device='cpu').numpy()
    expert_pos = expert_offsets_cpu.copy()
    for t in range(num_tokens):
        for k in range(num_topk):
            if topk_idx[t, k] < 0:
                token_topk_to_pos_cpu[t, k] = -1
                continue
            e = topk_idx[t, k].item()
            p = expert_pos[e]
            pos_to_expert_cpu[p] = e
            token_topk_to_pos_cpu[t, k] = p
            expert_pos[e] += 1
    pos_to_expert = torch.from_numpy(pos_to_expert_cpu).to(topk_idx.device)
    token_topk_to_pos = torch.from_numpy(token_topk_to_pos_cpu).to(topk_idx.device)
    return token_topk_to_pos, pos_to_expert, expert_offsets, aligned_expanded
