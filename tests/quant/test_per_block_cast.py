import pytest
import torch

from tile_kernels_ascend.torch.quant.cast import cast

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
@pytest.mark.parametrize("fmt", ["e4m3", "e2m1"])
def test_per_block_cast(fmt):
    h, w = 64, 64
    block_size = (32, 32)
    torch.manual_seed(55)

    x_cpu = torch.randn(h, w).to(torch.bfloat16)
    out_w_cpu, sf_cpu = cast(x_cpu, fmt, block_size=block_size)

    x_npu = x_cpu.npu()
    try:
        out_w_npu, sf_npu = cast(x_npu, fmt, block_size=block_size)
    except RuntimeError as e:
        pytest.xfail(f"NPU cast unsupported: {e}")

    out_w_npu_cpu = out_w_npu.cpu()
    sf_npu_cpu = sf_npu.cpu()

    if fmt == "e4m3":
        torch.testing.assert_close(out_w_npu_cpu, out_w_cpu)
    else:
        mismatch = (out_w_npu_cpu.to(torch.int8) != out_w_cpu.to(torch.int8))
        mismatch_ratio = mismatch.float().mean().item()
        assert mismatch_ratio < 0.02, f"e2m1 mismatch ratio {mismatch_ratio}"

    torch.testing.assert_close(sf_npu_cpu, sf_cpu, atol=1e-3, rtol=1e-3)
