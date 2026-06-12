"""
torch_npu.npu_swiglu_quant — Fused SwiGLU activation + quantization

What it does:
    Applies SwiGLU(x) followed by quantization in a single fused kernel.
    SwiGLU splits the input in half along the last axis and applies:
        - left activate:  swish(x_left) * x_right
        - right activate: x_left * swish(x_right)
    Then quantizes the result (per-channel / per-token, static / dynamic).

Common usage patterns:
    1. Static per-channel quant (quant_mode=0)  — used by
       swiglu_forward_and_per_channel_cast_and_transpose
    2. Dynamic per-token quant  (quant_mode=1)  — used by
       swiglu_forward_and_per_token_cast
    3. MoE grouped mode: multiple smooth_scales groups (group_index + group_list_type)
    4. Left vs right activation (activate_left=True / False)

Signature:
    torch_npu.npu_swiglu_quant(
        x, *,
        smooth_scales=None,    # [G, N] or [G] smooth factors
        offsets=None,          # [G, N] or [G] offsets (only for static quant_mode=0)
        group_index=None,      # [G] int32, cumsum or count of tokens per group
        activate_left=False,   # use left or right SiLU activation
        quant_mode=0,          # 0=static, 1=dynamic
        group_list_type=0,     # 0=cumsum, 1=count
        dst_type=None,         # None/int8=int8, int4/quint4x2=int4
    ) -> (out_tensor, scale_tensor)
"""

import torch
import torch_npu


def demo_static_per_channel_no_groups():
    """
    Pattern 1: static per-channel quant without grouping.
    quant_mode=0 means "static quantization" — smooth_scales are fixed
    (e.g. pre-calibrated per-channel scales) and offsets provide zero-points.

    Used by: swiglu_forward_and_per_channel_cast_and_transpose
    (which passes quant_mode=0, activate_left=True, no groups).
    """
    torch.manual_seed(0)
    num_tokens, hidden2 = 8, 64  # hidden2 = 2 * hidden, will be split in half
    N = hidden2 // 2
    x = torch.randn(num_tokens, hidden2, dtype=torch.bfloat16, device="npu")

    out, scale = torch_npu.npu_swiglu_quant(
        x,
        quant_mode=0,
        dst_type=torch.int8,
        activate_left=True,
    )

    assert out.dtype == torch.int8
    # SwiGLU halves the last dim: output has shape (num_tokens, hidden)
    assert out.shape == (num_tokens, N)
    print("[static/per-channel]  out.shape:", out.shape)


def demo_dynamic_per_token_no_groups():
    """
    Pattern 2: dynamic per-token quantization without groups.
    quant_mode=1 computes the per-token scale dynamically during execution.

    Used by: swiglu_forward_and_per_token_cast
    (which passes quant_mode=1, activate_left=True, no groups).
    """
    torch.manual_seed(1)
    num_tokens, hidden2 = 16, 128
    N = hidden2 // 2
    x = torch.randn(num_tokens, hidden2, dtype=torch.bfloat16, device="npu")

    out, scale = torch_npu.npu_swiglu_quant(
        x,
        quant_mode=1,
        dst_type=torch.int8,
        activate_left=True,
    )

    assert out.dtype == torch.int8
    assert out.shape == (num_tokens, N)
    # scale is per-token (same as npu_dynamic_quant)
    assert scale.dtype == torch.float32
    assert scale.shape == (num_tokens,)
    print("[dynamic/per-token]   out.shape:", out.shape, " scale.shape:", scale.shape)


def demo_dynamic_per_token_with_groups():
    """
    Pattern 3: MoE grouped dynamic per-token quant.
    Different groups of tokens use different smooth_scales.

    group_index uses cumsum semantics (group_list_type=0):
        group 0: tokens [0, group_index[0])          -> smooth_scales[0]
        group 1: tokens [group_index[0], group_index[1]) -> smooth_scales[1]
        ...

    Used in MoE forward where different expert "routes" share the same
    buffer but need per-group quantization scales (e.g. DeepSeek-MoE).
    """
    torch.manual_seed(2)
    num_tokens, hidden2 = 32, 128
    N = hidden2 // 2
    num_groups = 4
    group_size = num_tokens // num_groups

    x = torch.randn(num_tokens, hidden2, dtype=torch.float32, device="npu")
    smooth_scales = torch.randn(num_groups, N, dtype=torch.float32, device="npu")
    group_index = torch.arange(1, num_groups + 1, dtype=torch.int32, device="npu") * group_size

    out, scale = torch_npu.npu_swiglu_quant(
        x,
        smooth_scales=smooth_scales,
        offsets=None,
        group_index=group_index,
        quant_mode=1,
        group_list_type=0,
        activate_left=True,
        dst_type=torch.int8,
    )

    assert out.shape == (num_tokens, N)
    assert out.dtype == torch.int8
    assert scale.shape == (num_tokens,)
    print("[MoE grouped]         out.shape:", out.shape, " groups:", num_groups)


def demo_static_per_channel_with_groups():
    """
    Pattern 4: static per-channel quant with groups (quant_mode=0).
    In this mode, offsets ARE used (they provide the zero-point for static quant).
    Both smooth_scales and offsets have shape (num_groups, N).
    """
    torch.manual_seed(3)
    num_tokens, hidden2 = 24, 64
    N = hidden2 // 2
    num_groups = 3
    group_size = num_tokens // num_groups

    x = torch.randn(num_tokens, hidden2, dtype=torch.float32, device="npu")
    smooth_scales = torch.randn(num_groups, N, dtype=torch.float32, device="npu")
    offsets = torch.randn(num_groups, N, dtype=torch.float32, device="npu")
    group_index = torch.arange(1, num_groups + 1, dtype=torch.int32, device="npu") * group_size

    out, scale = torch_npu.npu_swiglu_quant(
        x,
        smooth_scales=smooth_scales,
        offsets=offsets,
        group_index=group_index,
        quant_mode=0,
        group_list_type=0,
        activate_left=False,
        dst_type=torch.int8,
    )

    assert out.shape == (num_tokens, N)
    assert out.dtype == torch.int8
    print("[static+groups]       out.shape:", out.shape)


def demo_group_list_type_count():
    """
    group_list_type=1 ("count" mode): group_index contains the number of
    tokens in each group (not cumulative sums).

    Same MoE semantics as cumsum mode, but easier to specify when
    each group processes the same number of tokens.
    """
    torch.manual_seed(4)
    num_tokens, hidden2 = 16, 64
    N = hidden2 // 2
    num_groups = 4
    group_size = num_tokens // num_groups

    x = torch.randn(num_tokens, hidden2, dtype=torch.float32, device="npu")
    smooth_scales = torch.randn(num_groups, N, dtype=torch.float32, device="npu")
    group_index = torch.full((num_groups,), group_size, dtype=torch.int32, device="npu")

    out, scale = torch_npu.npu_swiglu_quant(
        x,
        smooth_scales=smooth_scales,
        group_index=group_index,
        quant_mode=1,
        group_list_type=1,  # count mode
        activate_left=True,
        dst_type=torch.int8,
    )

    assert out.shape == (num_tokens, N)
    print("[count group_list]    out.shape:", out.shape)


def demo_activate_right():
    """
    activate_left=False (default) uses right-activation:
        SwiGLU(x) = x_left * swish(x_right)
    This is the "standard" SwiGLU from the Shazeer paper.
    """
    torch.manual_seed(5)
    num_tokens, hidden2 = 8, 64
    N = hidden2 // 2
    x = torch.randn(num_tokens, hidden2, dtype=torch.bfloat16, device="npu")

    out, scale = torch_npu.npu_swiglu_quant(
        x,
        quant_mode=1,
        dst_type=torch.int8,
        activate_left=False,  # right activation (standard SwiGLU)
    )

    assert out.shape == (num_tokens, N)
    assert out.dtype == torch.int8
    print("[right activation]    out.shape:", out.shape)


def demo_int4_output():
    """
    dst_type=torch.quint4x2 for INT4 SwiGLU-quant output.
    The last dim of input must be a multiple of 4.
    """
    torch.manual_seed(6)
    num_tokens, hidden2 = 8, 128  # 128 = 2*64, 64 is multiple of 4 -> OK
    N = hidden2 // 2
    x = torch.randn(num_tokens, hidden2, dtype=torch.float32, device="npu")

    out, scale = torch_npu.npu_swiglu_quant(
        x,
        quant_mode=1,
        dst_type=torch.quint4x2,
        activate_left=True,
    )

    # int4 is packed: output shape will reflect the packing
    assert scale.dtype == torch.float32
    print("[int4 output]         out.shape:", out.shape, " dtype:", out.dtype)


def demo_fp16_input():
    """
    FP16 input (instead of FP32 / BF16).
    All three input dtypes (fp16, bf16, fp32) are supported.
    """
    torch.manual_seed(7)
    num_tokens, hidden2 = 8, 64
    N = hidden2 // 2
    x = torch.randn(num_tokens, hidden2, dtype=torch.float16, device="npu")

    out, scale = torch_npu.npu_swiglu_quant(
        x, quant_mode=1, dst_type=torch.int8, activate_left=True
    )

    assert out.shape == (num_tokens, N)
    assert out.dtype == torch.int8
    print("[fp16 input]          out.shape:", out.shape)


if __name__ == "__main__":
    demo_static_per_channel_no_groups()
    demo_dynamic_per_token_no_groups()
    demo_dynamic_per_token_with_groups()
    demo_static_per_channel_with_groups()
    demo_group_list_type_count()
    demo_activate_right()
    demo_int4_output()
    demo_fp16_input()
    print("\nAll npu_swiglu_quant demos passed!")
