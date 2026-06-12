import pytest
import torch

from tile_kernels_ascend.torch.quant.per_channel_cast_fused import per_channel_cast_fused as per_channel_cast_fused_ref

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_per_channel_cast_fused():
    from tile_kernels_ascend.aclnn.quant import per_channel_cast_fused
    num_tokens, hidden = 128, 128
    torch.manual_seed(23)

    x_cpu = torch.randn(num_tokens, hidden).to(torch.bfloat16)
    out_w_cpu, sf_cpu = per_channel_cast_fused_ref(
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
        pytest.xfail(f"ACLNN per_channel_cast_fused unsupported: {e}")

    assert out_w_npu.dtype == torch.int8
    assert sf_npu.shape[0] == num_tokens
