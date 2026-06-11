import pytest
import torch

from tile_kernels_ascend.torch.engram import grad_w_reduce_ref
from tile_kernels_ascend.torch.utils import make_param_id


NUM_TOKENS = 256
HC_MULT = 4
HIDDEN = 512


@pytest.fixture
def grad_w_reduce_inputs():
    grad_w_partial = torch.randn(NUM_TOKENS, HC_MULT, HC_MULT, HIDDEN, dtype=torch.float32)
    weight_hidden = torch.randn(HC_MULT, HIDDEN, dtype=torch.bfloat16)
    weight_embed = torch.randn(HC_MULT, HIDDEN, dtype=torch.bfloat16)
    return grad_w_partial, weight_hidden, weight_embed


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_grad_w_reduce_npu_vs_cpu(grad_w_reduce_inputs):
    grad_w_partial, weight_hidden, weight_embed = grad_w_reduce_inputs

    golden = grad_w_reduce_ref(grad_w_partial, weight_hidden, weight_embed)

    result = grad_w_reduce_ref(
        grad_w_partial.to('npu'),
        weight_hidden.to('npu'),
        weight_embed.to('npu'),
    ).cpu()

    param_id = make_param_id({
        'num_tokens': NUM_TOKENS,
        'hidden': HIDDEN,
        'hc_mult': HC_MULT,
    })
    torch.testing.assert_close(result, golden, rtol=1e-3, atol=1e-3)
