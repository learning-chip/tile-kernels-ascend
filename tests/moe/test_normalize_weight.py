import pytest
import torch

from tile_kernels_ascend.torch.moe import normalize_weight


NUM_TOKENS = 128
NUM_TOPK = 4


@pytest.fixture
def normalize_weight_inputs():
    topk_weights = torch.rand(NUM_TOKENS, NUM_TOPK, dtype=torch.float32)
    return (topk_weights,)


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_normalize_weight_npu_vs_cpu(normalize_weight_inputs):
    topk_weights, = normalize_weight_inputs

    golden_denominator, golden_normalized = normalize_weight(topk_weights)

    result_denominator, result_normalized = normalize_weight(topk_weights.to('npu'))
    result_denominator = result_denominator.cpu()
    result_normalized = result_normalized.cpu()

    torch.testing.assert_close(result_denominator, golden_denominator, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result_normalized, golden_normalized, rtol=1e-5, atol=1e-5)
