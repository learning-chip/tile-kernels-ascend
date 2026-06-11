import pytest
import torch

from tile_kernels_ascend.torch.moe import get_fused_mapping
from tile_kernels_ascend.torch.moe.expand_to_fused import expand_to_fused, expand_to_fused_with_sf


NUM_TOKENS = 64
HIDDEN = 128
NUM_TOPK = 4
NUM_EXPERTS = 8
ALIGNMENT = 16


@pytest.fixture
def expand_to_fused_inputs():
    topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64)
    max_expanded = NUM_TOKENS * NUM_TOPK
    num_expanded_tokens = ((max_expanded + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    token_topk_to_pos, pos_to_expert, _, aligned_expanded = get_fused_mapping(
        num_expanded_tokens, NUM_TOKENS, NUM_TOPK, topk_idx, ALIGNMENT,
    )
    x = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16)
    return x, token_topk_to_pos, pos_to_expert, aligned_expanded


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_expand_to_fused_npu_vs_cpu(expand_to_fused_inputs):
    x, token_topk_to_pos, pos_to_expert, aligned_expanded = expand_to_fused_inputs

    golden = expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    result = expand_to_fused(
        x.to('npu'), token_topk_to_pos.to('npu'), pos_to_expert.to('npu'),
    ).cpu()

    torch.testing.assert_close(result, golden, rtol=0, atol=0)


@pytest.fixture
def expand_to_fused_sf_inputs():
    topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64)
    max_expanded = NUM_TOKENS * NUM_TOPK
    num_expanded_tokens = ((max_expanded + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    token_topk_to_pos, pos_to_expert, _, aligned_expanded = get_fused_mapping(
        num_expanded_tokens, NUM_TOKENS, NUM_TOPK, topk_idx, ALIGNMENT,
    )
    x_data = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16)
    hidden_sf = HIDDEN // 32
    x_sf = torch.randn(NUM_TOKENS, hidden_sf, dtype=torch.bfloat16)
    num_per_channels = 32
    return (x_data, x_sf), token_topk_to_pos, pos_to_expert, aligned_expanded, num_per_channels, hidden_sf


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_expand_to_fused_with_sf_npu_vs_cpu(expand_to_fused_sf_inputs):
    (x_data, x_sf), token_topk_to_pos, pos_to_expert, aligned_expanded, num_per_channels, hidden_sf = expand_to_fused_sf_inputs

    golden_out, golden_sf = expand_to_fused_with_sf(
        (x_data, x_sf), num_per_channels, token_topk_to_pos, pos_to_expert,
    )

    result_out, result_sf = expand_to_fused_with_sf(
        (x_data.to('npu'), x_sf.to('npu')),
        num_per_channels,
        token_topk_to_pos.to('npu'),
        pos_to_expert.to('npu'),
    )
    result_out = result_out.cpu()
    result_sf = result_sf.cpu()

    torch.testing.assert_close(result_out, golden_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(result_sf, golden_sf, rtol=1e-2, atol=1e-2)
