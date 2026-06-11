import pytest
import torch

from tile_kernels_ascend.torch.moe import mask_indices_by_tp


N = 128
NUM_EP_RANKS = 4
TP_RANK = 0
NUM_TP_RANKS = 2


@pytest.fixture
def mask_indices_inputs():
    per_gpu = N // NUM_EP_RANKS
    max_val = per_gpu * NUM_EP_RANKS * NUM_TP_RANKS
    indices = torch.randint(0, max_val, (N,), dtype=torch.int64)
    return (indices,)


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_mask_indices_by_tp_npu_vs_cpu(mask_indices_inputs):
    indices, = mask_indices_inputs

    golden = mask_indices_by_tp(indices, N, NUM_EP_RANKS, TP_RANK, NUM_TP_RANKS)

    result = mask_indices_by_tp(indices.to('npu'), N, NUM_EP_RANKS, TP_RANK, NUM_TP_RANKS).cpu()

    torch.testing.assert_close(result, golden, rtol=0, atol=0)
