import pytest
import torch

from tile_kernels_ascend.torch.moe import get_fused_mapping


NUM_TOKENS = 64
NUM_TOPK = 4
NUM_EXPERTS = 8
ALIGNMENT = 16


@pytest.fixture
def get_fused_mapping_inputs():
    topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64)
    max_expanded = NUM_TOKENS * NUM_TOPK
    num_expanded_tokens = ((max_expanded + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    return topk_idx, num_expanded_tokens


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_get_fused_mapping_npu_vs_cpu(get_fused_mapping_inputs):
    topk_idx, num_expanded_tokens = get_fused_mapping_inputs

    g_tok_topk, g_pos_exp, g_exp_off, g_aligned = get_fused_mapping(
        num_expanded_tokens, NUM_TOKENS, NUM_TOPK, topk_idx, ALIGNMENT,
    )

    r_tok_topk, r_pos_exp, r_exp_off, r_aligned = get_fused_mapping(
        num_expanded_tokens, NUM_TOKENS, NUM_TOPK, topk_idx.to('npu'), ALIGNMENT,
    )
    r_tok_topk = r_tok_topk.cpu()
    r_pos_exp = r_pos_exp.cpu()
    r_exp_off = r_exp_off.cpu()

    torch.testing.assert_close(r_tok_topk, g_tok_topk, rtol=0, atol=0)
    torch.testing.assert_close(r_pos_exp, g_pos_exp, rtol=0, atol=0)
    torch.testing.assert_close(r_exp_off, g_exp_off, rtol=0, atol=0)
    assert r_aligned == g_aligned
