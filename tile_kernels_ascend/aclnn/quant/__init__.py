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
    if fmt == 'e4m3':
        col_bs = x.shape[-1]
        if col_bs > 256:
            col_bs = 128
        return torch_npu.npu_dynamic_block_quant(
            x, row_block_size=1, col_block_size=col_bs, dst_type=292
        )
    return torch_npu.npu_dynamic_quant(x)


def per_channel_cast(
    x: torch.Tensor,
    fmt: str = 'e4m3',
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch_npu.npu_dynamic_quant(x, quant_mode="perchannel")


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
    if fmt == 'e2m1':
        raise NotImplementedError(
            "ACLNN backend: per_block_cast with e2m1 (FP4) not supported — "
            "npu_dynamic_block_quant only produces FP8/HiFLOAT8 output on Ascend 950"
        )
    row_bs, col_bs = block_size
    return torch_npu.npu_dynamic_block_quant(
        x, row_block_size=row_bs, col_block_size=col_bs, dst_type=292
    )


def per_block_cast_lossless(
    x: torch.Tensor,
    block_size: tuple[int, int] = (32, 32),
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "ACLNN backend: per_block_cast_lossless — row_block_size=32 not supported on Ascend 950 "
        "(only 1/128/256/512), and no FP8 anti_quant available"
    )


def cast_back(
    x: tuple[torch.Tensor, torch.Tensor],
    fmt: str = 'fp32',
    block_size: tuple[int, int] = (32, 32),
) -> torch.Tensor:
    x_data, x_sf = x
    out_dtype = torch.float32 if fmt == 'fp32' else torch.bfloat16
    if x_data.dtype == torch.float8_e4m3fn:
        x_float = x_data.to(torch.float32)
        row_bs, col_bs = block_size
        sf = x_sf
        if sf.dtype == torch.int32:
            sf = sf.contiguous()
            if sf.stride(-1) != 1:
                sf = sf.as_strided(size=sf.shape, stride=(sf.shape[-1], 1))
            sf = sf.view(torch.uint8).to(torch.int32)
            sf = (sf << 23).view(torch.float32)
        sf = sf.repeat_interleave(row_bs, dim=0).repeat_interleave(col_bs, dim=1)
        sf = sf[: x_data.shape[0], : x_data.shape[1]]
        return (x_float * sf).to(out_dtype)
    if x_sf.dim() > 1:
        x_sf_flat = x_sf[..., 0]
    else:
        x_sf_flat = x_sf
    return torch_npu.npu_anti_quant(x_data, x_sf_flat, dst_dtype=out_dtype if out_dtype != torch.float32 else torch.float16)


def cast_back_e5m6(
    x: tuple[torch.Tensor, torch.Tensor],
    fmt: str = 'fp32',
    block_size: tuple[int, int] = (32, 32),
) -> torch.Tensor:
    raise NotImplementedError(
        "ACLNN backend: cast_back_e5m6 — e5m6 custom float format not available in torch_npu"
    )


def swiglu_forward_and_per_channel_cast_and_transpose(
    x: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch_npu.npu_swiglu_quant(
        x, quant_mode=0, dst_type=torch.int8, activate_left=True
    )


def swiglu_forward_and_per_token_cast(
    x: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch_npu.npu_swiglu_quant(
        x, quant_mode=1, dst_type=torch.int8, activate_left=True
    )


def swiglu_backward_and_per_token_cast(
    x: tuple[torch.Tensor, torch.Tensor],
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    **kwargs,
) -> tuple:
    raise NotImplementedError(
        "ACLNN backend: swiglu_backward_and_per_token_cast — no SwiGLU backward API in torch_npu"
    )


def per_token_cast_to_e5m6(
    x: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "ACLNN backend: per_token_cast_to_e5m6 (e5m6 custom float format) not available in torch_npu"
    )
