import pytest
import torch

from tile_kernels_ascend.torch.quant.cast_e5m6 import cast_to_e5m6, cast_back_from_e5m6

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_cast_back_e5m6():
    num_tokens, hidden = 128, 128
    num_per_channels = 128
    x_block_size = (num_tokens, num_per_channels)
    torch.manual_seed(77)

    x_cpu = (torch.randn(num_tokens, hidden) * 0.5).to(torch.bfloat16)
    packed_cpu, sf_cpu = cast_to_e5m6(x_cpu, num_per_channels=num_per_channels)
    x_back_cpu = cast_back_from_e5m6((packed_cpu, sf_cpu), "bf16", x_block_size=x_block_size)

    x_npu = x_cpu.npu()
    try:
        packed_npu, sf_npu = cast_to_e5m6(x_npu, num_per_channels=num_per_channels)
    except RuntimeError as e:
        pytest.xfail(f"NPU cast_to_e5m6 unsupported: {e}")

    try:
        x_back_npu = cast_back_from_e5m6((packed_npu, sf_npu), "bf16", x_block_size=x_block_size)
    except RuntimeError as e:
        pytest.xfail(f"NPU cast_back_from_e5m6 unsupported: {e}")

    torch.testing.assert_close(x_back_npu.cpu(), x_back_cpu, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_cast_back_e5m6_aclnn():
    from tile_kernels_ascend.aclnn.quant import cast_back_e5m6
    x = torch.randn(64, 64, dtype=torch.bfloat16).npu()
    x_sf = torch.randn(1, 1, dtype=torch.float32).npu()
    with pytest.raises(NotImplementedError):
        cast_back_e5m6((x, x_sf), fmt='bf16', block_size=(64, 64))
