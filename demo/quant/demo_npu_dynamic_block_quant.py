"""
torch_npu.npu_dynamic_block_quant — Block-wise dynamic quantization

What it does:
    Divides the input into (row_block_size x col_block_size) blocks and
    quantizes each block independently:
        block_max = max(abs(x_block))
        scale     = DTYPE_MAX / block_max
        y_block   = round(x_block * scale)
    Returns (y_quantized, scale_per_block).

Common usage patterns:
    1. FP8  e4m3 per-token  (1, hidden)   — per-token FP8 quant (used by per_token_cast)
    2. FP8  e4m3 per-block  (R, C)        — tile-wise FP8 quant (used by per_block_cast)
    3. FP8  e4m3 large hidden             — multi-chunk per-token when hidden > 128

dst_type values:
    292   = FP8 e4m3fn           (torch.float8_e4m3fn, primary usage on Ascend 950)
    Other values (HiFLOAT8, FP8 e5m2) exist but the enum IDs are version-specific.

On Ascend 950, only FP8-type outputs (FP8_E4M3FN, FP8_E5M2, HiFLOAT8) are
supported — INT8 output is NOT available for npu_dynamic_block_quant.
Only 2-D inputs are supported; 3-D is not accepted.

Signature:
    torch_npu.npu_dynamic_block_quant(
        x, *, min_scale=0.0, round_mode="rint",
        dst_type=1, row_block_size=1, col_block_size=128,
    ) -> (Tensor, Tensor)
"""

import torch
import torch_npu


def demo_fp8_per_token():
    """
    Pattern 1: FP8 e4m3 per-token quant (row_block_size=1, col_block_size=hidden).
    Used by per_token_cast(fmt='e4m3') in tile-kernels-ascend.
    Each row is quantized independently to FP8 e4m3fn format.

    dst_type=292 selects FP8 e4m3fn output (same as torch.float8_e4m3fn).
    """
    torch.manual_seed(1)
    num_tokens, hidden = 128, 128
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="npu")

    out, scale = torch_npu.npu_dynamic_block_quant(
        x, row_block_size=1, col_block_size=hidden, dst_type=292
    )

    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == x.shape
    assert scale.shape == (num_tokens, 1)
    print("[FP8 per-token]    out.dtype:", out.dtype, " scale.shape:", scale.shape)


def demo_fp8_per_block():
    """
    Pattern 2: FP8 e4m3 per-block quant (e.g. block_size=(128, 128)).
    Used by per_block_cast(block_size=(128, 128), fmt='e4m3') in tile-kernels-ascend.
    Each (128 x 128) tile is quantized independently.
    """
    torch.manual_seed(2)
    h, w = 256, 256
    row_bs, col_bs = 128, 128
    x = torch.randn(h, w, dtype=torch.bfloat16, device="npu")

    out, scale = torch_npu.npu_dynamic_block_quant(
        x, row_block_size=row_bs, col_block_size=col_bs, dst_type=292
    )

    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == (h, w)
    assert scale.shape == (h // row_bs, w // col_bs)
    print(f"[FP8 per-block({row_bs},{col_bs})]  out.shape:", out.shape, " scale.shape:", scale.shape)


def demo_fp8_per_token_large_hidden():
    """
    Pattern 3: When hidden > 256, per_token_cast caps col_block_size at 128.
    This becomes per-(token, 128-chunk) quant, with multiple scales per token.
    """
    torch.manual_seed(3)
    num_tokens, hidden = 16, 512
    col_bs = min(hidden, 128)
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="npu")

    out, scale = torch_npu.npu_dynamic_block_quant(
        x, row_block_size=1, col_block_size=col_bs, dst_type=292
    )

    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == (num_tokens, hidden)
    assert scale.shape == (num_tokens, hidden // col_bs)
    print(f"[FP8 large hid={hidden}] col_bs={col_bs}  scale.shape:", scale.shape)


def demo_fp8_row_block_size_1_col_128():
    """
    Pattern 4: the most common (and currently only supported on Ascend 950)
    configuration — row_block_size=1, col_block_size=128.
    Works with any 2-D input whose width is a multiple of 128.
    """
    torch.manual_seed(4)
    x = torch.randn(64, 128, dtype=torch.float16, device="npu")

    out, scale = torch_npu.npu_dynamic_block_quant(
        x, row_block_size=1, col_block_size=128, dst_type=292
    )

    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == x.shape
    assert scale.shape == (64, 1)
    print("[FP8 (1,128) fp16] out.shape:", out.shape, " scale.shape:", scale.shape)


def demo_fp8_min_scale():
    """
    Pattern 5: min_scale parameter — a lower bound on the computed scale,
    used to avoid overflow when block values are very small.
    scale = min(FP8_MAX / block_max,  1 / min_scale)  when min_scale > 0
    """
    torch.manual_seed(5)
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="npu") * 0.001

    out, scale = torch_npu.npu_dynamic_block_quant(
        x, min_scale=1e-3, row_block_size=1, col_block_size=128, dst_type=292
    )

    assert out.dtype == torch.float8_e4m3fn
    print("[FP8 min_scale]   scale min bounded, min scale:", scale.min().item())


def demo_hifloat8_output():
    """
    HiFLOAT8 output — an 8-bit float with larger exponent range
    than standard FP8 e4m3fn. Supported on Ascend 950.
    The dst_type value for HiFLOAT8 is device/version specific; check
    your CANN version to find the correct enum value.
    """
    torch.manual_seed(6)
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="npu")

    try:
        out, scale = torch_npu.npu_dynamic_block_quant(
            x, row_block_size=1, col_block_size=128, dst_type=129
        )
        assert out.shape == x.shape
        print("[HiFLOAT8]        out.shape:", out.shape, " out.dtype:", out.dtype)
    except (RuntimeError, ValueError):
        print("[HiFLOAT8]        SKIPPED (dst_type=129 not supported in current CANN)")


def demo_fp8_e5m2_output():
    """
    FP8 e5m2 output (dst_type=130) — 5-bit exponent, 2-bit mantissa.
    Greater dynamic range but less precision than e4m3fn.
    Note: support for this dst_type value varies across CANN versions.
    """
    torch.manual_seed(7)
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="npu")

    try:
        out, scale = torch_npu.npu_dynamic_block_quant(
            x, row_block_size=1, col_block_size=128, dst_type=130
        )
        assert out.shape == x.shape
        print("[FP8 e5m2]        out.shape:", out.shape, " out.dtype:", out.dtype)
    except (RuntimeError, ValueError):
        print("[FP8 e5m2]        SKIPPED (dst_type=130 not supported in current CANN)")


if __name__ == "__main__":
    demo_fp8_per_token()
    demo_fp8_per_block()
    demo_fp8_per_token_large_hidden()
    demo_fp8_row_block_size_1_col_128()
    demo_fp8_min_scale()
    demo_hifloat8_output()
    demo_fp8_e5m2_output()
    print("\nAll npu_dynamic_block_quant demos passed!")
