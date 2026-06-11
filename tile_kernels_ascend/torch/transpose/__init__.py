import torch


def transpose_ref(x: torch.Tensor) -> torch.Tensor:
    return x.T.contiguous()


def batched_transpose_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.transpose(x, 1, 2).contiguous()
