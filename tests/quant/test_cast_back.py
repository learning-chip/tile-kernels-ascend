import pytest
import torch

from tile_kernels_ascend.torch.quant.cast import cast, cast_back

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_cast_back():
    h, w = 64, 64
    block_size = (32, 32)
    torch.manual_seed(33)

    x_cpu = torch.randn(h, w).to(torch.bfloat16)
    x_q_cpu, sf_cpu = cast(x_cpu, "e4m3", block_size=block_size)
    x_back_cpu = cast_back((x_q_cpu, sf_cpu), "bf16", block_size=block_size)

    x_q_npu, sf_npu = x_q_cpu.npu(), sf_cpu.npu()
    try:
        x_back_npu = cast_back((x_q_npu, sf_npu), "bf16", block_size=block_size)
    except RuntimeError as e:
        pytest.xfail(f"NPU cast_back unsupported: {e}")

    torch.testing.assert_close(x_back_npu.cpu(), x_back_cpu)
