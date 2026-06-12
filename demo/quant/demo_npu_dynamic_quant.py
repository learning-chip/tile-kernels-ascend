"""
torch_npu.npu_dynamic_quant — Per-token / per-channel dynamic quantization

What it does:
    Quantizes a float tensor (fp16 / bf16) row-by-row. For each row it computes:
        scale = max(abs(x_row * smooth_scales)) / DTYPE_MAX
        y     = round(x_row * smooth_scales / scale)
    Returns (y_int, scale_float).

Common usage patterns:
    1. Basic per-token INT8 quantization   (default, no extra args)
    2. Per-channel INT8 quantization       (quant_mode="perchannel", Ascend950+)
    3. Smooth-quant scaling                (smooth_scales=[hidden])
    4. MoE grouped smooth-quant scaling    (smooth_scales=[G, hidden] + group_index)
    5. INT4 output (quint4x2)              (dst_type=torch.quint4x2)

Signature:
    torch_npu.npu_dynamic_quant(
        x, *, smooth_scales=None, group_index=None,
        dst_type=None, quant_mode="pertoken",
    ) -> (Tensor, Tensor)
"""

import torch
import torch_npu


def demo_per_token_int8_fp16():
    """
    Pattern 1: basic per-token INT8 quant from FP16 input.
    This is the simplest call — just quantize each row of x to int8.
    Used in per_channel_cast_fused / per_token_cast fallback paths.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float16, device="npu")

    out, scale = torch_npu.npu_dynamic_quant(x)

    assert out.dtype == torch.int8
    assert out.shape == x.shape
    assert scale.dtype == torch.float32
    # scale shape: x.shape with last dim removed
    assert scale.shape == x.shape[:-1]
    print("[per-token INT8 fp16]  out.shape:", out.shape, " scale.shape:", scale.shape)


def demo_per_token_int8_bf16():
    """
    Same as above but with BF16 input — commonly used on Ascend 910B / 950
    because BF16 gives better numerics for training workloads.
    """
    torch.manual_seed(1)
    x = torch.randn(4, 32, dtype=torch.bfloat16, device="npu")

    out, scale = torch_npu.npu_dynamic_quant(x)

    assert out.dtype == torch.int8
    assert scale.dtype == torch.float32
    print("[per-token INT8 bf16]  out.shape:", out.shape, " scale.dtype:", scale.dtype)


def demo_smooth_quant():
    """
    Pattern 3: smooth-quant — multiply each channel by a learned smooth factor
    before quantization. This redistributes quant error across channels for
    better accuracy in LLM-style models.
    smooth_scales shape = (hidden,) — one scale per channel.
    """
    torch.manual_seed(2)
    num_tokens, hidden = 8, 64
    x = torch.randn(num_tokens, hidden, dtype=torch.float16, device="npu")
    smooth_scales = torch.randn(hidden, dtype=torch.float16, device="npu")

    out, scale = torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales)

    assert out.shape == x.shape
    # scale is per-token
    assert scale.shape == (num_tokens,)
    print("[smooth-quant]         out.shape:", out.shape, " scale.shape:", scale.shape)


def demo_moe_grouped_smooth_quant():
    """
    Pattern 4: MoE grouped smooth-quant — different expert groups use different
    smooth_scales. group_index is a cumsum of per-group token counts.

    Example layout (3 groups, 12 tokens total, hidden=32):
        group 0: tokens [0:4)  -> smooth_scales[0]
        group 1: tokens [4:8)  -> smooth_scales[1]
        group 2: tokens [8:12) -> smooth_scales[2]
    group_index = [4, 8, 12]  (cumulative token counts)
    """
    torch.manual_seed(3)
    num_tokens, hidden = 12, 32
    num_groups = 3
    x = torch.randn(num_tokens, hidden, dtype=torch.float16, device="npu")
    # smooth_scales: one row per group, one col per channel
    smooth_scales = torch.randn(num_groups, hidden, dtype=torch.float16, device="npu")
    # group_index: cumulative token counts, last element = num_tokens
    group_index = torch.tensor([4, 8, 12], dtype=torch.int32, device="npu")

    out, scale = torch_npu.npu_dynamic_quant(
        x, smooth_scales=smooth_scales, group_index=group_index
    )

    assert out.shape == x.shape
    assert scale.shape == (num_tokens,)
    print("[moe-grouped smooth]   out.shape:", out.shape, " scale.shape:", scale.shape)


def demo_per_channel_int8():
    """
    Pattern 2: per-channel INT8 quantization (Ascend 950 / Atlas A3+).
    Instead of per-row (per-token), quantize per-column (per-channel).
    Used in per_channel_cast in tile-kernels-ascend.

    This is equivalent to computing the scale along the token axis rather
    than the hidden axis — useful for weight-style quantization where
    channels share a common dynamic range.
    """
    torch.manual_seed(4)
    num_tokens, hidden = 128, 128
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="npu")

    out, scale = torch_npu.npu_dynamic_quant(x, quant_mode="perchannel")

    assert out.dtype == torch.int8
    assert out.shape == x.shape
    # per-channel: scale has one element per channel (last dim)
    assert scale.shape == (hidden,)
    print("[per-channel INT8]     out.shape:", out.shape, " scale.shape:", scale.shape)


def demo_per_channel_smooth():
    """
    Per-channel quant combined with per-channel smooth scales.
    smooth_scales shape = (hidden, 1) in caller code but here
    the API expects a 1-D vector of size hidden.
    """
    torch.manual_seed(5)
    num_tokens, hidden = 64, 64
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="npu")
    smooth_scales = torch.randn(hidden, dtype=torch.bfloat16, device="npu")

    out, scale = torch_npu.npu_dynamic_quant(
        x, smooth_scales=smooth_scales, quant_mode="perchannel"
    )

    assert out.dtype == torch.int8
    assert scale.shape == (hidden,)
    print("[per-channel+smooth]   out.shape:", out.shape, " scale.shape:", scale.shape)


def demo_int4_quant():
    """
    Pattern 5: INT4 output (quint4x2).
    Each output byte packs two int4 values.
    Requires x's last dim to be a multiple of 8.
    Returns y with shape [..., last_dim/8] and dtype int32 (8 int4s per int32).

    Note: INT4 quant via npu_dynamic_quant is only supported on Ascend 910B.
    On Ascend 950 this call will raise a RuntimeError.
    """
    torch.manual_seed(6)
    num_tokens, hidden = 4, 64
    x = torch.randn(num_tokens, hidden, dtype=torch.float16, device="npu")

    try:
        out, scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)
        assert out.dtype == torch.int32
        assert out.shape == (num_tokens, hidden // 8)
        print("[int4 quint4x2]        out.shape:", out.shape, " out.dtype:", out.dtype)
    except RuntimeError:
        print("[int4 quint4x2]        SKIPPED (not supported on Ascend 950)")


def demo_3d_input():
    """
    3-D input (batch, seq, hidden) works the same way on Ascend 910B —
    quantization is applied along the last dim for each row in the flattened
    leading dims.

    Note: on Ascend 950 the default pertoken mode may not support 3-D inputs;
    use npu_dynamic_block_quant for block-wise quant on 3-D tensors instead.
    """
    torch.manual_seed(7)
    x = torch.randn(2, 16, 64, dtype=torch.bfloat16, device="npu")

    try:
        out, scale = torch_npu.npu_dynamic_quant(x)
        assert out.shape == x.shape
        assert scale.shape == (2, 16)
        print("[3-D bf16 input]       out.shape:", out.shape, " scale.shape:", scale.shape)
    except RuntimeError:
        print("[3-D bf16 input]       SKIPPED (not supported on Ascend 950)")


if __name__ == "__main__":
    demo_per_token_int8_fp16()
    demo_per_token_int8_bf16()
    demo_smooth_quant()
    demo_moe_grouped_smooth_quant()
    demo_per_channel_int8()
    demo_per_channel_smooth()
    demo_int4_quant()
    demo_3d_input()
    print("\nAll npu_dynamic_quant demos passed!")
