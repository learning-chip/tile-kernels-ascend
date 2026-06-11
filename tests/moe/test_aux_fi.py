import pytest
import torch

from tile_kernels_ascend.torch.moe import aux_fi


NUM_TOKENS = 128
NUM_TOPK = 4
NUM_EXPERTS = 32
NUM_AUX_TOPK = 2


@pytest.fixture
def aux_fi_inputs():
    topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64)
    return (topk_idx,)


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_aux_fi_npu_vs_cpu(aux_fi_inputs):
    topk_idx, = aux_fi_inputs

    golden = aux_fi(topk_idx, NUM_EXPERTS, NUM_AUX_TOPK)

    result = aux_fi(topk_idx.to('npu'), NUM_EXPERTS, NUM_AUX_TOPK).cpu()

    torch.testing.assert_close(result, golden, rtol=0, atol=0)
