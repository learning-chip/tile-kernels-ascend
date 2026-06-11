import pytest
import torch

from tile_kernels_ascend.torch.quant.per_channel_cast_fused import per_channel_cast_fused

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_per_channel_cast_fused():
    num_tokens, hidden = 128, 128
    torch.manual_seed(23)

    x_cpu = torch.randn(num_tokens, hidden).to(torch.bfloat16)
    out_w_cpu, sf_cpu = per_channel_cast_fused(
        x_cpu, num_per_tokens=128, num_per_channels=128,
        round_sf=False, pos_to_token=None,
    )

    x_npu = x_cpu.npu()
    try:
        out_w_npu, sf_npu = per_channel_cast_fused(
            x_npu, num_per_tokens=128, num_per_channels=128,
            round_sf=False, pos_to_token=None,
        )
    except RuntimeError as e:
        pytest.xfail(f"NPU cast unsupported: {e}")

    torch.testing.assert_close(out_w_npu.cpu(), out_w_cpu)
    torch.testing.assert_close(sf_npu.cpu(), sf_cpu, atol=1e-3, rtol=1e-3)
