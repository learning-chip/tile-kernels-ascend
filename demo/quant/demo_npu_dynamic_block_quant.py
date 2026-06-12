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
    1. INT8 per-row-block (1, 128)  — row-wise INT8 quant
    2. FP8  e4m3 per-token (1, N)   — per-token FP8 quant (used by per_token_cast)
    3. FP8  e4m3 per-block (R, C)   — tile-wise FP8 quant (used by per_block_cast)

dst_type values:
    1   = INT8       (default)
    292 = FP8 e4m3fn (torch.float8_e4m3fn)

Signature:
    torch_npu.npu_dynamic_block_quant(
        x, *, min_scale=0.0, round_mode="rint",
        dst_type=1, row_block_size=1, col_block_size=128,
    ) -> (Tensor, Tensor)

Constraints (on Ascend 950):
    - row_block_size: 1 (most common), 128, or other powers of 2
    - col_block_size: usually 128
    - x must be 2-D or 3-D
"""

import torch
import torch_npu


def demo_int8_per_row_block():
    """
    Pattern 1: INT8 per-row block quant with col_block_size=128.
    Each (1 x 128) block gets its own scale.
    If the last dim is not divisible by 128, the API pads internally.
    """
    torch.manual_seed(0)
    rows, cols = 8, 256  # cols must be divisible by 128
    x = torch.randn(rows, cols, dtype=torch.float16, device="npu")

    out, scale = torch_npu.npu_dynamic_block_quant(
        x, row_block_size=1, col_block_size=128, dst_type=1
    )

    assert out.dtype == torch.int8
    assert out.shape == x.shape
    # one scale per (row, block-of-128-cols)
    assert scale.shape == (rows, cols // 128)
    print("[INT8 (1,128)]    out.shape:", out.shape, " scale.shape:", scale.shape)


def demo_fp8_per_token():
    """
    Pattern 2: FP8 e4m3 per-token quant (row_block_size=1, col_block_size=hidden).
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
    print("[FP8 per-token]   out.dtype:", out.dtype, " scale.shape:", scale.shape)


def demo_fp8_per_block():
    """
    Pattern 3: FP8 e4m3 per-block quant (e.g. block_size=(128, 128)).
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
    # scale shape = (h/row_bs, w/col_bs)
    assert scale.shape == (h // row_bs, w // col_bs)
    print(f"[FP8 per-block({row_bs},{col_bs})]  out.shape:", out.shape, " scale.shape:", scale.shape)


def demo_fp8_per_token_large_hidden():
    """
    When hidden > 256, per_token_cast caps col_block_size at 128
    (to avoid exceeding hardware limits). This effectively becomes
    per-(token, 128-chunk) quant, with multiple scales per token.
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
    # multiple scale chunks per token
    assert scale.shape == (num_tokens, hidden // col_bs)
    print(f"[FP8 large hid={hidden}] col_bs={col_bs}  scale.shape:", scale.shape)


def demo_3d_input_int8():
    """
    3-D input (batch, seq, hidden) is flattened over leading dims:
    treated as (batch*seq, hidden) internally, then reshaped back.
    """
    torch.manual_seed(4)
    x = torch.randn(2, 16, 128, dtype=torch.float16, device="npu")

    out, scale = torch_npu.npu_dynamic_block_quant(
        x, row_block_size=1, col_block_size=128, dst_type=1
    )

    assert out.dtype == torch.int8
    assert out.shape == x.shape
    print("[3D INT8]       out.shape:", out.shape, " scale.shape:", scale.shape)


def demo_min_scale():
    """
    min_scale parameter: a lower bound on the computed scale, used to
    avoid division by very small numbers (and thus overflowing quantized values).
    scale = min(DTYPE_MAX / block_max,  1 / min_scale)  when min_scale > 0
    """
    torch.manual_seed(5)
    x = torch.randn(4, 128, dtype=torch.float16, device="npu") * 0.001  # tiny values

    out, scale = torch_npu.npu_dynamic_block_quant(
        x, min_scale=1e-3, row_block_size=1, col_block_size=128, dst_type=1
    )

    assert out.dtype == torch.int8
    print("[min_scale]     scale min bounded, min scale:", scale.min().item())


if __name__ == "__main__":
    demo_int8_per_row_block()
    demo_fp8_per_token()
    demo_fp8_per_block()
    demo_fp8_per_token_large_hidden()
    demo_3d_input_int8()
    demo_min_scale()
    print("\nAll npu_dynamic_block_quant demos passed!")
