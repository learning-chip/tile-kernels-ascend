import torch
from typing import Optional, Union

try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    torch_npu = None
    NPU_AVAILABLE = False


def per_token_cast(
    x: torch.Tensor,
    fmt: str = 'e4m3',
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch_npu.npu_dynamic_quant(x)


def per_channel_cast(
    x: torch.Tensor,
    fmt: str = 'e4m3',
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch_npu.npu_dynamic_quant(x)


def per_channel_cast_and_transpose(
    x: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "ACLNN backend: per_channel_cast_and_transpose fused kernel not yet available in torch_npu"
    )


def per_channel_cast_fused(
    x: Union[torch.Tensor, tuple],
    num_per_tokens: int,
    num_per_channels: Optional[int],
    round_sf: bool,
    pos_to_token: Optional[torch.Tensor],
) -> tuple:
    if isinstance(x, tuple):
        x = x[0]
    if pos_to_token is not None:
        x = x[pos_to_token.clamp(min=0)]
    return torch_npu.npu_dynamic_quant(x)


def per_block_cast(
    x: torch.Tensor,
    block_size: tuple[int, int] = (1, 128),
    fmt: str = 'e4m3',
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_bs, col_bs = block_size
    return torch_npu.npu_dynamic_block_quant(x, row_block_size=row_bs, col_block_size=col_bs)


def per_block_cast_lossless(
    x: torch.Tensor,
    block_size: tuple[int, int] = (32, 32),
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "ACLNN backend: per_block_cast_lossless not yet available in torch_npu"
    )


def cast_back(
    x: tuple[torch.Tensor, torch.Tensor],
    fmt: str = 'fp32',
    block_size: tuple[int, int] = (32, 32),
) -> torch.Tensor:
    x_data, x_sf = x
    if x_sf.dim() > 1:
        x_sf = x_sf[..., 0]
    out_dtype = torch.float32 if fmt == 'fp32' else torch.bfloat16
    return torch_npu.npu_anti_quant(x_data, x_sf, dst_dtype=out_dtype)


def cast_back_e5m6(
    x: tuple[torch.Tensor, torch.Tensor],
    fmt: str = 'fp32',
    block_size: tuple[int, int] = (32, 32),
) -> torch.Tensor:
    raise NotImplementedError(
        "ACLNN backend: cast_back_e5m6 not yet available in torch_npu"
    )


def swiglu_forward_and_per_channel_cast_and_transpose(
    x: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "ACLNN backend: swiglu_forward_and_per_channel_cast_and_transpose fused kernel not yet available"
    )


def swiglu_forward_and_per_token_cast(
    x: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "ACLNN backend: swiglu_forward_and_per_token_cast fused kernel not yet available"
    )


def swiglu_backward_and_per_token_cast(
    x: tuple[torch.Tensor, torch.Tensor],
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    **kwargs,
) -> tuple:
    raise NotImplementedError(
        "ACLNN backend: swiglu_backward_and_per_token_cast fused kernel not yet available"
    )


def per_token_cast_to_e5m6(
    x: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "ACLNN backend: per_token_cast_to_e5m6 (e5m6 custom float format) not yet available in torch_npu"
    )
