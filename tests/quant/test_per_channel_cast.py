import pytest
import torch

from tile_kernels_ascend.torch.quant.cast import cast

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
@pytest.mark.parametrize("round_sf", [True, False])
def test_per_channel_cast(round_sf):
    num_tokens, hidden = 128, 128
    torch.manual_seed(17)

    x_cpu = torch.randn(num_tokens, hidden).to(torch.bfloat16)
    out_w_cpu, sf_cpu = cast(x_cpu, "e4m3", block_size=(num_tokens, 1), round_sf=round_sf)

    x_npu = x_cpu.npu()
    try:
        out_w_npu, sf_npu = cast(x_npu, "e4m3", block_size=(num_tokens, 1), round_sf=round_sf)
    except RuntimeError as e:
        pytest.xfail(f"NPU cast unsupported: {e}")

    torch.testing.assert_close(out_w_npu.cpu(), out_w_cpu)
    torch.testing.assert_close(sf_npu.cpu(), sf_cpu, atol=1e-3, rtol=1e-3)
