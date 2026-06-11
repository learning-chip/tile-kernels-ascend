from typing import Iterable
import torch


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def dtype_to_str(dtype: torch.dtype) -> str:
    mapping = {
        torch.float32: 'fp32',
        torch.bfloat16: 'bf16',
        torch.float8_e4m3fn: 'e4m3',
        torch.int8: 'e2m1',
    }
    if dtype not in mapping:
        raise ValueError(f'Unsupported dtype: {dtype}')
    return mapping[dtype]


def make_param_id(params: dict) -> str:
    parts = []
    for key, value in params.items():
        if isinstance(value, torch.dtype):
            parts.append(f'{key}={dtype_to_str(value)}')
        elif isinstance(value, tuple):
            parts.append(f'{key}={"x".join(str(v) for v in value)}')
        else:
            parts.append(f'{key}={value}')
    return '-'.join(parts) if parts else 'default'


def generate_num_tokens(alignment: int = 1) -> list[int]:
    return [align(t, alignment) for t in [4001, 8001]]


def generate_hidden_sizes(align_val: int = 64) -> list[int]:
    return [h for h in [576, 2048, 2560, 3072, 4096, 6144, 7168] if h % align_val == 0]


def generate_moe_params() -> Iterable[dict]:
    for num_tokens in (4001,):
        for num_topk in (2, 6, 8, 9):
            for num_experts in (72, 256):
                for num_ep_ranks in (8, 64):
                    if num_experts % num_ep_ranks == 0:
                        yield {
                            'num_send_tokens': num_tokens,
                            'num_topk': num_topk,
                            'num_experts': num_experts // num_ep_ranks,
                            'num_ep_ranks': num_ep_ranks,
                        }


def generate_topk_idx(params: dict, device: str = 'cpu') -> torch.Tensor:
    num_send_tokens = params['num_send_tokens']
    num_experts = params['num_experts']
    num_topk = params['num_topk']
    num_ep_ranks = params['num_ep_ranks']
    if num_send_tokens == 0:
        return torch.empty((0, num_topk), dtype=torch.int64, device=device)
    scores = torch.rand(
        (num_send_tokens * num_ep_ranks, num_experts * num_ep_ranks),
        dtype=torch.bfloat16, device=device,
    )
    _, topk_idx = torch.topk(scores, k=num_topk, dim=-1, sorted=False)
    mask = topk_idx >= num_experts
    topk_idx[mask] = -1
    mask = mask.all(dim=1)
    topk_idx = topk_idx[~mask]
    return topk_idx
