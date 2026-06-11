import pytest
import torch

from tile_kernels_ascend.torch.moe import topk_sum_and_topk_group_idx


NUM_TOKENS = 64
NUM_GROUPS = 4
NUM_GROUP_SUM_TOPK = 2
NUM_TOPK_GROUPS = 2


@pytest.fixture
def topk_sum_inputs():
    num_per_group = 4
    scores = torch.randn(NUM_TOKENS, NUM_GROUPS, num_per_group, dtype=torch.float32)
    return (scores,)


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_topk_sum_and_topk_group_idx_npu_vs_cpu(topk_sum_inputs):
    scores, = topk_sum_inputs

    golden = topk_sum_and_topk_group_idx(scores, NUM_GROUP_SUM_TOPK, NUM_TOPK_GROUPS)

    result = topk_sum_and_topk_group_idx(scores.to('npu'), NUM_GROUP_SUM_TOPK, NUM_TOPK_GROUPS).cpu()

    torch.testing.assert_close(result, golden, rtol=0, atol=0)
