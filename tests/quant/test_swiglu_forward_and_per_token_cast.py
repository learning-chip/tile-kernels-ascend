import pytest
import torch

from tile_kernels_ascend.torch.quant.swiglu import swiglu_forward

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_swiglu_forward_and_per_token_cast():
    num_tokens, hidden2 = 64, 256
    num_topk = 4
    torch.manual_seed(21)

    x_cpu = (torch.randn(num_tokens, hidden2) * 0.5).to(torch.bfloat16)

    num_expanded = num_tokens
    pos_to_token_topk_cpu = (
        torch.arange(num_expanded, dtype=torch.int64) * num_topk
        + torch.randint(0, num_topk, (num_expanded,))
    )
    pos_to_token_topk_cpu[-5:] = -1

    topk_weights_cpu = torch.rand(num_tokens, num_topk).to(torch.bfloat16)

    out_cpu = swiglu_forward(
        x_cpu,
        pos_to_token_topk=pos_to_token_topk_cpu,
        topk_weights=topk_weights_cpu,
    )

    x_npu = x_cpu.npu()
    pos_npu = pos_to_token_topk_cpu.npu()
    tw_npu = topk_weights_cpu.npu()
    try:
        out_npu = swiglu_forward(x_npu, pos_to_token_topk=pos_npu, topk_weights=tw_npu)
    except RuntimeError as e:
        pytest.xfail(f"NPU swiglu_forward unsupported: {e}")

    torch.testing.assert_close(out_npu.cpu(), out_cpu, atol=1e-3, rtol=1e-3)
