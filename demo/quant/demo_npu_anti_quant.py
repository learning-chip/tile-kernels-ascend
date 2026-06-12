"""
torch_npu.npu_anti_quant — De-quantization (int -> float)

What it does:
    Reverses quantization by applying:  out = (x + offset) * scale
    This is the inverse of npu_dynamic_quant / npu_dynamic_block_quant.

Common usage patterns:
    1. INT8  -> FP16   (basic dequant, no offset)
    2. INT8  -> BF16   (for training / mixed-precision pipelines)
    3. INT8  + offset  -> FP16   (asymmetric dequant, useful with zero-point)
    4. INT32 (packed INT4) -> FP16 / BF16   (for INT4 weight dequant)

Signature:
    torch_npu.npu_anti_quant(
        x, scale, *, offset=None,
        dst_dtype=torch.float16, src_dtype=None,
    ) -> Tensor

Notes:
    - scale / offset must be 1-D, same dtype (fp32/bf16), shape (n,)
      where n == x.shape[-1] (or 8x for packed int4).
    - Ascend 950 (Atlas A3) only — not supported on 910B for all dtypes.
"""

import torch
import torch_npu


def demo_int8_to_fp16():
    """
    Pattern 1: basic INT8 -> FP16 dequantization.
    scale shape = (hidden,) matches the channel (last) dim of x.
    """
    torch.manual_seed(0)
    num_tokens, hidden = 8, 32
    x = torch.randint(-127, 127, (num_tokens, hidden), dtype=torch.int8, device="npu")
    scale = torch.randn(hidden, dtype=torch.float32, device="npu")

    out = torch_npu.npu_anti_quant(x, scale, dst_dtype=torch.float16)

    assert out.dtype == torch.float16
    assert out.shape == x.shape
    print("[int8->fp16]   out.shape:", out.shape, " dtype:", out.dtype)


def demo_int8_to_bf16():
    """
    Pattern 2: INT8 -> BF16 — commonly used in BF16 training pipelines.
    """
    torch.manual_seed(1)
    num_tokens, hidden = 8, 32
    x = torch.randint(-127, 127, (num_tokens, hidden), dtype=torch.int8, device="npu")
    scale = torch.randn(hidden, dtype=torch.float32, device="npu")

    out = torch_npu.npu_anti_quant(x, scale, dst_dtype=torch.bfloat16)

    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape
    print("[int8->bf16]   out.shape:", out.shape, " dtype:", out.dtype)


def demo_asymmetric_dequant():
    """
    Pattern 3: asymmetric dequant (with offset / zero-point).
    Used when the quantization used a zero-point, i.e. the original
    quant formula was:  y = round((x - zero) / scale)
    Dequant then is:      x_hat = (y + zero) * scale
    offset shape must match scale shape.
    """
    torch.manual_seed(2)
    num_tokens, hidden = 4, 16
    x = torch.randint(-127, 127, (num_tokens, hidden), dtype=torch.int8, device="npu")
    scale = torch.abs(torch.randn(hidden, dtype=torch.float32, device="npu")) + 0.01
    offset = torch.randn(hidden, dtype=torch.float32, device="npu") * 10

    out = torch_npu.npu_anti_quant(x, scale, offset=offset, dst_dtype=torch.float16)

    assert out.dtype == torch.float16
    assert out.shape == x.shape
    print("[asymmetric]   out.shape:", out.shape, " offset.shape:", offset.shape)


def test_roundtrip_per_channel():
    """
    Full round-trip with per-channel quantization.
    npu_anti_quant expects scale with shape (hidden,) — matching the last dim of x.
    This matches the per-channel scale convention of npu_dynamic_quant.
    """
    torch.manual_seed(3)
    num_tokens, hidden = 4, 32
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="npu")

    # Forward: BF16 -> INT8 + per-channel scale
    x_q, scale = torch_npu.npu_dynamic_quant(x, quant_mode="perchannel")
    assert x_q.dtype == torch.int8
    # scale shape = (hidden,) for perchannel mode
    assert scale.shape == (hidden,)

    # Reverse: INT8 + scale -> BF16
    x_hat = torch_npu.npu_anti_quant(x_q, scale, dst_dtype=torch.bfloat16)

    assert x_hat.dtype == torch.bfloat16
    assert x_hat.shape == x.shape

    err = (x.float() - x_hat.float()).abs().max().item()
    assert err < 0.5, f"Roundtrip error too large: {err}"
    print(f"[roundtrip per-ch]  max abs err: {err:.4f}")


def demo_src_dtype_hint():
    """
    Passing src_dtype=torch.int8 explicitly.
    Useful when the API signature needs the source dtype disambiguated
    (e.g., when x.dtype is ambiguous after a view/cast).
    """
    torch.manual_seed(4)
    x = torch.randint(-100, 100, (4, 16), dtype=torch.int8, device="npu")
    scale = torch.randn(16, dtype=torch.float32, device="npu")

    out = torch_npu.npu_anti_quant(
        x, scale, dst_dtype=torch.float16, src_dtype=torch.int8
    )
    assert out.dtype == torch.float16
    print("[src_dtype]    explicitly passed src_dtype=torch.int8")


if __name__ == "__main__":
    demo_int8_to_fp16()
    demo_int8_to_bf16()
    demo_asymmetric_dequant()
    test_roundtrip_per_channel()
    demo_src_dtype_hint()
    print("\nAll npu_anti_quant demos passed!")
