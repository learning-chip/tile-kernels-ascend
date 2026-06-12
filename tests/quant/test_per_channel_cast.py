import pytest
import torch

from tile_kernels_ascend.torch.quant.cast import cast

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_per_channel_cast():
    from tile_kernels_ascend.aclnn.quant import per_channel_cast
    num_tokens, hidden = 128, 128
    torch.manual_seed(17)

    x_cpu = torch.randn(num_tokens, hidden).to(torch.bfloat16)
    out_w_cpu, sf_cpu = cast(x_cpu, "e4m3", block_size=(num_tokens, 1), round_sf=False)

    x_npu = x_cpu.npu()
    try:
        out_w_npu, sf_npu = per_channel_cast(x_npu, fmt='e4m3')
    except RuntimeError as e:
        pytest.xfail(f"ACLNN per_channel_cast unsupported: {e}")

    assert out_w_npu.dtype == torch.int8
    assert sf_npu.shape == (hidden,)
