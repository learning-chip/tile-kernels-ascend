import torch


def transpose(x: torch.Tensor) -> torch.Tensor:
    return x.t().contiguous()


def batched_transpose(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-2, -1).contiguous()
