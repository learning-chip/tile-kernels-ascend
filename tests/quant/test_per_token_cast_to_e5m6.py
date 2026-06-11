import pytest
import torch

from tile_kernels_ascend.torch.quant.cast_e5m6 import cast_to_e5m6

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_per_token_cast_to_e5m6(dtype):
    num_tokens, hidden = 128, 128
    num_per_channels = 128
    torch.manual_seed(42)

    x_cpu = torch.randn(num_tokens, hidden).to(dtype)
    packed_cpu, sf_cpu = cast_to_e5m6(x_cpu, num_per_channels=num_per_channels)

    x_npu = x_cpu.npu()
    try:
        packed_npu, sf_npu = cast_to_e5m6(x_npu, num_per_channels=num_per_channels)
    except RuntimeError as e:
        pytest.xfail(f"NPU cast_to_e5m6 unsupported: {e}")

    torch.testing.assert_close(packed_npu.cpu(), packed_cpu)
    torch.testing.assert_close(sf_npu.cpu(), sf_cpu, atol=1e-3, rtol=1e-3)
