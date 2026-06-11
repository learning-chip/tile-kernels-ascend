import torch


def make_offsets(vocab_sizes: torch.Tensor) -> torch.Tensor:
    num_ngram_layers = vocab_sizes.shape[0]
    offsets_list = []
    for layer_idx in range(num_ngram_layers):
        flat = vocab_sizes[layer_idx].view(-1)
        prefix = torch.cat([torch.zeros(1, dtype=torch.int32, device=flat.device), flat[:-1].cumsum(0, dtype=torch.int32)])
        offsets_list.append(prefix)
    return torch.stack(offsets_list, dim=0)


def engram_hash_ref(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    num_ngram_layers = multipliers.shape[0]
    max_ngram_size = multipliers.shape[1]
    prod = ngram_token_ids.to(torch.int64).unsqueeze(0) * multipliers.unsqueeze(1)
    ans = [[] for _ in range(num_ngram_layers)]
    hashes = prod[:, :, 0].clone()
    for i in range(1, max_ngram_size):
        hashes.bitwise_xor_(prod[:, :, i])
        for layer_idx in range(num_ngram_layers):
            ans[layer_idx].append(
                (hashes[layer_idx].unsqueeze(-1) % vocab_sizes[layer_idx, i - 1].to(torch.int64).unsqueeze(0)).to(torch.int32)
            )
    for layer_idx in range(num_ngram_layers):
        ans[layer_idx] = torch.cat(ans[layer_idx], dim=-1)
    output = torch.stack(ans, dim=0)
    return output + offsets.unsqueeze(1)


def engram_gate_ref(
    hidden_states: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
    clamp_value: float,
    eps: float,
    save_for_backward: bool = False,
):
    hidden_size = hidden_states.shape[-1]
    scalar = hidden_size ** -0.5
    x = hidden_states.float()
    k_f = k.float()
    wh = weight_hidden.float().unsqueeze(0)
    we = weight_embed.float().unsqueeze(0)
    rstd_x = torch.rsqrt(x.pow(2).mean(-1) + eps)
    rstd_k = torch.rsqrt(k_f.pow(2).mean(-1) + eps)
    raw_dot = torch.einsum('...d,...d->...', x * wh, k_f * we)
    dot = raw_dot * rstd_x * rstd_k * scalar
    signed_sqrt = dot.abs().clamp_min(clamp_value).sqrt() * dot.sign()
    gate_score = signed_sqrt.sigmoid()
    output = x + gate_score.unsqueeze(-1) * v.unsqueeze(-2)
    output = output.bfloat16()
    if save_for_backward:
        return output, raw_dot, gate_score, rstd_x, rstd_k
    return output


def grad_w_reduce_ref(
    grad_w_partial: torch.Tensor,
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
) -> torch.Tensor:
    return (grad_w_partial * weight_hidden.unsqueeze(0).unsqueeze(2)).sum(dim=-2) + \
           (grad_w_partial * weight_embed.unsqueeze(0).unsqueeze(2)).sum(dim=-2)


def fused_weight_ref(
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
) -> torch.Tensor:
    return (weight_hidden * weight_embed).float()
