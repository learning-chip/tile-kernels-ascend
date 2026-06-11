import os
import pytest
import torch

os.environ.setdefault('TORCHINDUCTOR_DISABLE', '1')

from tile_kernels_ascend.torch.moe import get_fused_mapping
from tile_kernels_ascend.torch.moe import reduce_fused as _reduce_mod
from tile_kernels_ascend.torch.moe.reduce_fused import reduce_fused


NUM_TOKENS = 64
HIDDEN = 128
NUM_TOPK = 4
NUM_EXPERTS = 8
ALIGNMENT = 16


@pytest.fixture(autouse=True)
def _disable_compile():
    _reduce_mod.elementwise_fma = lambda a, b, c: a * b + c
    yield


@pytest.fixture
def reduce_fused_inputs():
    topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64)
    max_expanded = NUM_TOKENS * NUM_TOPK
    num_expanded_tokens = ((max_expanded + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    token_topk_to_pos, pos_to_expert, _, aligned_expanded = get_fused_mapping(
        num_expanded_tokens, NUM_TOKENS, NUM_TOPK, topk_idx, ALIGNMENT,
    )
    x = torch.randn(aligned_expanded, HIDDEN, dtype=torch.bfloat16)
    topk_weights = torch.rand(NUM_TOKENS, NUM_TOPK, dtype=torch.float32)
    return x, topk_weights, token_topk_to_pos, aligned_expanded


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_reduce_fused_npu_vs_cpu(reduce_fused_inputs):
    x, topk_weights, token_topk_to_pos, aligned_expanded = reduce_fused_inputs

    golden = reduce_fused(x, topk_weights, token_topk_to_pos)

    result = reduce_fused(
        x.to('npu'), topk_weights.to('npu'), token_topk_to_pos.to('npu'),
    ).cpu()

    torch.testing.assert_close(result, golden, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_reduce_fused_no_weights_npu_vs_cpu(reduce_fused_inputs):
    x, _, token_topk_to_pos, aligned_expanded = reduce_fused_inputs

    golden = reduce_fused(x, None, token_topk_to_pos)

    result = reduce_fused(x.to('npu'), None, token_topk_to_pos.to('npu')).cpu()

    torch.testing.assert_close(result, golden, rtol=1e-2, atol=1e-2)
