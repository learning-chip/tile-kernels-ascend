import pytest
import torch

from tile_kernels_ascend.torch.moe import group_count


NUM_GROUPS = 8


@pytest.fixture
def group_count_inputs():
    num_total_groups = 64
    group_idx = torch.randint(0, NUM_GROUPS, (num_total_groups,), dtype=torch.int64)
    return (group_idx,)


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_group_count_npu_vs_cpu(group_count_inputs):
    group_idx, = group_count_inputs

    golden = group_count(group_idx, NUM_GROUPS)

    result = group_count(group_idx.to('npu'), NUM_GROUPS).cpu()

    torch.testing.assert_close(result, golden, rtol=0, atol=0)
