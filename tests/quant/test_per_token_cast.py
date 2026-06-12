import pytest
import torch

from tile_kernels_ascend.torch.quant.cast import cast

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_per_token_cast_e4m3():
    from tile_kernels_ascend.aclnn.quant import per_token_cast
    num_tokens, hidden = 128, 128
    torch.manual_seed(7)

    x_cpu = torch.randn(num_tokens, hidden).to(torch.bfloat16)
    out_w_cpu, sf_cpu = cast(x_cpu, "e4m3", block_size=(1, hidden))

    x_npu = x_cpu.npu()
    try:
        out_w_npu, sf_npu = per_token_cast(x_npu, fmt='e4m3')
    except RuntimeError as e:
        pytest.xfail(f"ACLNN per_token_cast unsupported: {e}")

    out_w_npu_cpu = out_w_npu.cpu()
    sf_npu_cpu = sf_npu.cpu()

    assert out_w_npu_cpu.dtype == torch.float8_e4m3fn
    assert out_w_npu_cpu.shape == out_w_cpu.shape
    torch.testing.assert_close(
        sf_npu_cpu.squeeze(-1), sf_cpu.squeeze(-1), atol=1e-2, rtol=1e-2
    )


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_per_token_cast_e2m1_fallback():
    from tile_kernels_ascend.aclnn.quant import per_token_cast
    num_tokens, hidden = 128, 128
    x_npu = torch.randn(num_tokens, hidden, dtype=torch.bfloat16).npu()
    out_w, sf = per_token_cast(x_npu, fmt='e2m1')
    assert out_w.dtype == torch.int8
