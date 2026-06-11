import pytest
import torch

from tile_kernels_ascend.torch.moe import inplace_unique_group_indices


NUM_TOKENS = 64
NUM_TOPK = 4
NUM_GROUPS = 8


@pytest.fixture
def inplace_unique_inputs():
    group_indices = torch.randint(0, NUM_GROUPS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64)
    return (group_indices,)


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_inplace_unique_group_indices_npu_vs_cpu(inplace_unique_inputs):
    group_indices, = inplace_unique_inputs

    golden = group_indices.clone()
    inplace_unique_group_indices(golden, NUM_GROUPS)

    result = group_indices.clone().to('npu')
    inplace_unique_group_indices(result, NUM_GROUPS)
    result = result.cpu()

    torch.testing.assert_close(result, golden, rtol=0, atol=0)
