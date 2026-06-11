import pytest
import torch

from tile_kernels_ascend.torch.engram import fused_weight_ref
from tile_kernels_ascend.torch.utils import make_param_id


HC_MULT = 4
HIDDEN = 512


@pytest.fixture
def fused_weight_inputs():
    weight_hidden = torch.randn(HC_MULT, HIDDEN, dtype=torch.bfloat16)
    weight_embed = torch.randn(HC_MULT, HIDDEN, dtype=torch.bfloat16)
    return weight_hidden, weight_embed


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_fused_weight_npu_vs_cpu(fused_weight_inputs):
    weight_hidden, weight_embed = fused_weight_inputs

    golden = fused_weight_ref(weight_hidden, weight_embed)

    result = fused_weight_ref(
        weight_hidden.to('npu'),
        weight_embed.to('npu'),
    ).cpu()

    param_id = make_param_id({
        'hidden': HIDDEN,
        'hc_mult': HC_MULT,
        'dtype': torch.bfloat16,
    })
    torch.testing.assert_close(result, golden, rtol=1e-2, atol=1e-2)
