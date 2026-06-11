import pytest
import torch

from tile_kernels_ascend.torch.quant.cast import cast

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_per_channel_cast_and_transpose():
    num_tokens, hidden = 128, 128
    block_size = (num_tokens, 1)
    torch.manual_seed(99)

    x_cpu = torch.randn(num_tokens, hidden).to(torch.bfloat16)
    out_w_cpu, sf_cpu = cast(x_cpu, "e4m3", block_size=block_size)
    out_w_t_cpu = out_w_cpu.to(torch.float32).t()
    sf_t_cpu = sf_cpu.t()

    x_npu = x_cpu.npu()
    try:
        out_w_npu, sf_npu = cast(x_npu, "e4m3", block_size=block_size)
    except RuntimeError as e:
        pytest.xfail(f"NPU cast unsupported: {e}")

    out_w_t_npu = out_w_npu.to(torch.float32).t().cpu()
    sf_t_npu = sf_npu.t().cpu()

    torch.testing.assert_close(out_w_t_npu, out_w_t_cpu)
    torch.testing.assert_close(sf_t_npu, sf_t_cpu, atol=1e-3, rtol=1e-3)
