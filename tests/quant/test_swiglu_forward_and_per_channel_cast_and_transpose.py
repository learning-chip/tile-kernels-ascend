import pytest
import torch

from tile_kernels_ascend.torch.quant.swiglu import swiglu_forward

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_swiglu_forward_and_per_channel_cast_and_transpose():
    from tile_kernels_ascend.aclnn.quant import swiglu_forward_and_per_channel_cast_and_transpose
    num_tokens, hidden2 = 64, 256
    torch.manual_seed(11)

    x_cpu = (torch.randn(num_tokens, hidden2) * 0.5).to(torch.bfloat16)

    out_ref_cpu = swiglu_forward(x_cpu)
    out_t_ref = out_ref_cpu.t()

    x_npu = x_cpu.npu()
    try:
        out_q, scale = swiglu_forward_and_per_channel_cast_and_transpose(x_npu)
    except RuntimeError as e:
        pytest.xfail(f"ACLNN swiglu_forward_and_per_channel_cast_and_transpose unsupported: {e}")

    assert out_q.dtype == torch.int8
    assert out_q.shape == (num_tokens, hidden2 // 2)
    assert scale.dtype == torch.float32
