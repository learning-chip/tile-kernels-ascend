from typing import Optional, Tuple
import torch
from torch.types import Number
from tile_kernels_ascend.torch.quant.types import QuantTensor


@torch.compile
def elementwise_fma(a: torch.Tensor, b: Number | torch.Tensor, c: Number | torch.Tensor) -> torch.Tensor:
    return a * b + c


def swiglu_forward(
    x: torch.Tensor,
    pos_to_token_topk: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    swiglu_clamp_value: Optional[float] = None,
    clamped_count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert x.dim() == 2 and x.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float32)
    num_expanded_tokens, hidden2 = x.shape
    assert hidden2 % 2 == 0
    hidden = hidden2 // 2
    if pos_to_token_topk is not None:
        assert pos_to_token_topk.dim() == 1
        assert pos_to_token_topk.shape[0] == num_expanded_tokens
        assert topk_weights is not None
        assert topk_weights.dim() == 2
    x_fp32 = x.float()
    x_left = x_fp32[:, :hidden]
    x_right = x_fp32[:, hidden:]
    if swiglu_clamp_value is not None:
        if clamped_count is not None:
            clamped_count[0] += (x_left > swiglu_clamp_value).sum()
            clamped_count[1] += (x_right > swiglu_clamp_value).sum()
            clamped_count[2] += (x_right < -swiglu_clamp_value).sum()
        x_left = torch.clamp(x_left, max=swiglu_clamp_value)
        x_right = torch.clamp(x_right, min=-swiglu_clamp_value, max=swiglu_clamp_value)
    out = x_left / (1.0 + torch.exp(-x_left)) * x_right
    if pos_to_token_topk is not None:
        num_tokens, num_topk = topk_weights.shape
        pos_mask = pos_to_token_topk >= 0
        token_indices = torch.div(pos_to_token_topk[pos_mask], num_topk, rounding_mode='floor')
        topk_indices = pos_to_token_topk[pos_mask] % num_topk
        w_expanded = torch.zeros(num_expanded_tokens, device=x.device, dtype=torch.float32)
        w_expanded[pos_mask] = topk_weights[token_indices, topk_indices].float()
        out = out * w_expanded.unsqueeze(1)
    return out


def swiglu_backward(
    x: QuantTensor,
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    pos_to_token_topk: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    num_per_channels: int,
    swiglu_clamp_value: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_data, x_sf = x
    assert num_per_channels in (32, 128)
    assert (x_data.dim() == 2 or x_data.dim() == 3) and x_data.is_contiguous()
    assert x_sf.dim() == 2 and x_sf.is_contiguous()
    assert weight.dim() == 2 and weight.is_contiguous()
    assert pos_to_token_topk.dim() == 1
    assert token_topk_to_pos.dim() == 2 and token_topk_to_pos.is_contiguous()
    assert x_data.size(-1) % (2 * num_per_channels) == 0
    hidden = x_data.size(-1) // 2
    x_data = x_data.view(-1, hidden * 2)
    grad_out = grad_out.view(-1, hidden)
    num_expand_tokens = x_data.size(0)
    num_tokens, num_topk = token_topk_to_pos.shape
    assert x_sf.shape == (num_expand_tokens, 2 * hidden // num_per_channels)
    assert grad_out.shape == (num_expand_tokens, hidden)
    assert weight.shape == (num_tokens, num_topk)
    assert pos_to_token_topk.shape == (num_expand_tokens,)
    assert token_topk_to_pos.shape == (num_tokens, num_topk)
    x_sf_expanded = x_sf.repeat_interleave(num_per_channels, dim=1)
    x_fp32 = x_data.float() * x_sf_expanded
    x_part = x_fp32[:, :hidden]
    y_part = x_fp32[:, hidden:]
    use_clamp = swiglu_clamp_value is not None
    clamp_value = swiglu_clamp_value
    x_clamped = None
    y_clamped = None
    if use_clamp:
        x_clamped = x_part > clamp_value
        x_part[x_clamped] = clamp_value
        y_clamped_upper = y_part > clamp_value
        y_clamped_lower = y_part < -clamp_value
        y_clamped = y_clamped_upper | y_clamped_lower
        y_part[y_clamped_upper] = clamp_value
        y_part[y_clamped_lower] = -clamp_value
    tmp_x = 1.0 + torch.exp(-x_part)
    sigmoid_x = torch.ones_like(x_part) / tmp_x
    pos_mask = pos_to_token_topk >= 0
    token_indices = torch.div(pos_to_token_topk[pos_mask], num_topk, rounding_mode='floor')
    topk_indices = pos_to_token_topk[pos_mask] % num_topk
    w_expanded = torch.zeros(num_expand_tokens, device=x_data.device, dtype=torch.float32)
    w_expanded[pos_mask] = weight[token_indices, topk_indices]
    grad_out_fp32 = grad_out.float()
    grad_out_ws = grad_out_fp32 * w_expanded.unsqueeze(1) * sigmoid_x
    x_grad = grad_out_ws * y_part * elementwise_fma(x_part, 1.0 - sigmoid_x, 1.0)
    y_grad = grad_out_ws * x_part
    if use_clamp:
        x_grad[x_clamped] = 0.0
        y_grad[y_clamped] = 0.0
    act_out = x_part / tmp_x * y_part
    out = act_out * w_expanded.unsqueeze(1)
    x_grad_full = torch.cat([x_grad, y_grad], dim=1)
    weight_grad = torch.zeros_like(weight)
    dot_products = (grad_out_fp32 * act_out).sum(dim=1)
    weight_grad[token_indices, topk_indices] = dot_products[pos_mask]
    return out, x_grad_full, weight_grad
