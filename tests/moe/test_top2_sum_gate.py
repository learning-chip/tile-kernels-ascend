import pytest
import torch

from tile_kernels_ascend.torch.moe import top2_sum_gate


NUM_TOKENS = 32
NUM_ROUTED_EXPERTS = 16
NUM_TOPK = 2
NUM_GROUPS = 4
NUM_TOPK_GROUPS = 2


@pytest.fixture
def top2_sum_gate_inputs():
    logits = torch.randn(NUM_TOKENS, NUM_ROUTED_EXPERTS, dtype=torch.float32)
    bias = torch.randn(NUM_ROUTED_EXPERTS, dtype=torch.float32)
    return logits, bias


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_top2_sum_gate_npu_vs_cpu(top2_sum_gate_inputs):
    logits, bias = top2_sum_gate_inputs

    golden_idx, golden_weights = top2_sum_gate(
        logits=logits,
        bias=bias,
        num_topk=NUM_TOPK,
        num_topk_groups=NUM_TOPK_GROUPS,
        num_groups=NUM_GROUPS,
        use_shared_as_routed=False,
        num_shared_experts=0,
        routed_scaling_factor=1.0,
        ep_rank=0,
        num_ep_ranks=1,
        tp_rank=0,
        num_tp_ranks=1,
        scoring_func='softmax',
    )

    result_idx, result_weights = top2_sum_gate(
        logits=logits.to('npu'),
        bias=bias.to('npu'),
        num_topk=NUM_TOPK,
        num_topk_groups=NUM_TOPK_GROUPS,
        num_groups=NUM_GROUPS,
        use_shared_as_routed=False,
        num_shared_experts=0,
        routed_scaling_factor=1.0,
        ep_rank=0,
        num_ep_ranks=1,
        tp_rank=0,
        num_tp_ranks=1,
        scoring_func='softmax',
    )
    result_idx = result_idx.cpu()
    result_weights = result_weights.cpu()

    torch.testing.assert_close(result_idx, golden_idx, rtol=0, atol=0)
    torch.testing.assert_close(result_weights, golden_weights, rtol=1e-5, atol=1e-5)
