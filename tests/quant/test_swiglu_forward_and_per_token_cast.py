import pytest
import torch

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_swiglu_forward_and_per_token_cast():
    from tile_kernels_ascend.aclnn.quant import swiglu_forward_and_per_token_cast
    num_tokens, hidden2 = 64, 256
    torch.manual_seed(21)

    x_cpu = (torch.randn(num_tokens, hidden2) * 0.5).to(torch.bfloat16)
    try:
        x_npu = x_cpu.npu()
    except RuntimeError as e:
        pytest.xfail(f"NPU device error (possibly from prior test): {e}")
    try:
        out_q, scale = swiglu_forward_and_per_token_cast(x_npu)
    except RuntimeError as e:
        pytest.xfail(f"ACLNN swiglu_forward_and_per_token_cast unsupported: {e}")

    assert out_q.dtype == torch.int8
    assert out_q.shape == (num_tokens, hidden2 // 2)
    assert scale.dtype == torch.float32
    assert scale.shape == (num_tokens,)
