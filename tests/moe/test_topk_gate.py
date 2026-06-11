import pytest
import torch
import torch.nn.functional as F

from tile_kernels_ascend.torch.moe import topk_gate_ref


NUM_TOKENS = 128
NUM_EXPERTS = 32
NUM_TOPK = 4
SCORING_FUNCS = ['sigmoid', 'softmax', 'sqrtsoftplus']


@pytest.fixture
def topk_gate_inputs():
    logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32)
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32)
    return logits, bias


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
@pytest.mark.parametrize('scoring_func', SCORING_FUNCS)
def test_topk_gate_npu_vs_cpu(topk_gate_inputs, scoring_func):
    logits, bias = topk_gate_inputs

    golden_idx, golden_weights = topk_gate_ref(logits, bias, NUM_TOPK, scoring_func)

    result_idx, result_weights = topk_gate_ref(
        logits.to('npu'), bias.to('npu'), NUM_TOPK, scoring_func,
    )
    result_idx = result_idx.cpu()
    result_weights = result_weights.cpu()

    torch.testing.assert_close(result_idx, golden_idx, rtol=0, atol=0)
    torch.testing.assert_close(result_weights, golden_weights, rtol=1e-5, atol=1e-5)
