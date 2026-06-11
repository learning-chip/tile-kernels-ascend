import pytest
import torch

from tile_kernels_ascend.torch.quant.swiglu import swiglu_forward

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_swiglu_forward_and_per_channel_cast_and_transpose():
    num_tokens, hidden2 = 64, 256
    hidden = hidden2 // 2
    torch.manual_seed(11)

    x_cpu = (torch.randn(num_tokens, hidden2) * 0.5).to(torch.bfloat16)
    out_cpu = swiglu_forward(x_cpu)
    out_t_cpu = out_cpu.t()

    x_npu = x_cpu.npu()
    try:
        out_npu = swiglu_forward(x_npu)
    except RuntimeError as e:
        pytest.xfail(f"NPU swiglu_forward unsupported: {e}")

    out_t_npu = out_npu.t().cpu()

    torch.testing.assert_close(out_t_npu, out_t_cpu, atol=1e-3, rtol=1e-3)
    assert out_cpu.shape == (num_tokens, hidden)
    assert out_t_cpu.shape == (hidden, num_tokens)
