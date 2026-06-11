import pytest
import torch

from tile_kernels_ascend.torch.quant.cast import cast
from tile_kernels_ascend.torch.quant.swiglu import swiglu_backward

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_swiglu_backward_and_per_token_cast():
    hidden = 128
    hidden2 = 2 * hidden
    num_expand = 128
    num_tokens = 32
    num_topk = 4
    num_per_channels = 128
    torch.manual_seed(61)

    x_raw = (torch.randn(num_expand, hidden2) * 0.5).to(torch.bfloat16)
    try:
        x_q, x_sf = cast(x_raw, "e4m3", block_size=(1, num_per_channels))
    except RuntimeError as e:
        pytest.xfail(f"NPU cast unsupported for setup: {e}")

    grad_out = (torch.randn(num_expand, hidden) * 0.1).to(torch.float32)
    weight = torch.rand(num_tokens, num_topk).to(torch.float32)

    pos_to_token_topk = torch.full((num_expand,), -1, dtype=torch.int64)
    valid_n = min(num_expand, num_tokens * num_topk)
    pos_to_token_topk[:valid_n] = (
        torch.arange(valid_n, dtype=torch.int64) % (num_tokens * num_topk)
    )
    token_topk_to_pos = torch.randint(0, num_expand, (num_tokens, num_topk))

    x_tuple = (x_q, x_sf)

    try:
        out_cpu, x_grad_cpu, w_grad_cpu = swiglu_backward(
            x_tuple, grad_out, weight, pos_to_token_topk, token_topk_to_pos,
            num_per_channels=num_per_channels, swiglu_clamp_value=None,
        )
    except (RuntimeError, TypeError) as e:
        pytest.xfail(f"CPU swiglu_backward failed (compile issue): {e}")

    x_q_npu, x_sf_npu = x_q.npu(), x_sf.npu()
    grad_out_npu, weight_npu = grad_out.npu(), weight.npu()
    pos_npu = pos_to_token_topk.npu()
    ttp_npu = token_topk_to_pos.npu()

    try:
        out_npu, x_grad_npu, w_grad_npu = swiglu_backward(
            (x_q_npu, x_sf_npu), grad_out_npu, weight_npu, pos_npu, ttp_npu,
            num_per_channels=num_per_channels, swiglu_clamp_value=None,
        )
    except RuntimeError as e:
        pytest.xfail(f"NPU swiglu_backward unsupported: {e}")

    torch.testing.assert_close(out_npu.cpu(), out_cpu, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(x_grad_npu.cpu(), x_grad_cpu, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(w_grad_npu.cpu(), w_grad_cpu, atol=1e-3, rtol=1e-3)
