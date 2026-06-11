import pytest
import torch

from tile_kernels_ascend.torch.engram import engram_gate_ref
from tile_kernels_ascend.torch.utils import make_param_id


NUM_TOKENS = 256
HC_MULT = 4
HIDDEN = 512
CLAMP_VALUE = 0.0
EPS = 1e-6


@pytest.fixture
def engram_gate_inputs():
    hidden_states = torch.randn(NUM_TOKENS, HC_MULT, HIDDEN, dtype=torch.bfloat16)
    k = torch.randn(NUM_TOKENS, HC_MULT, HIDDEN, dtype=torch.bfloat16)
    v = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16)
    weight_hidden = torch.randn(HC_MULT, HIDDEN, dtype=torch.bfloat16)
    weight_embed = torch.randn(HC_MULT, HIDDEN, dtype=torch.bfloat16)
    return hidden_states, k, v, weight_hidden, weight_embed


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_engram_gate_fwd_no_save(engram_gate_inputs):
    hidden_states, k, v, weight_hidden, weight_embed = engram_gate_inputs

    golden = engram_gate_ref(
        hidden_states, k, v, weight_hidden, weight_embed,
        CLAMP_VALUE, EPS, save_for_backward=False,
    )

    result = engram_gate_ref(
        hidden_states.to('npu'), k.to('npu'), v.to('npu'),
        weight_hidden.to('npu'), weight_embed.to('npu'),
        CLAMP_VALUE, EPS, save_for_backward=False,
    ).cpu()

    param_id = make_param_id({
        'num_tokens': NUM_TOKENS,
        'hidden': HIDDEN,
        'hc_mult': HC_MULT,
        'dtype': torch.bfloat16,
    })
    torch.testing.assert_close(result, golden, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_engram_gate_fwd_save(engram_gate_inputs):
    hidden_states, k, v, weight_hidden, weight_embed = engram_gate_inputs

    golden_out, golden_raw_dot, golden_gate, golden_rstd_x, golden_rstd_k = engram_gate_ref(
        hidden_states, k, v, weight_hidden, weight_embed,
        CLAMP_VALUE, EPS, save_for_backward=True,
    )

    result_out, result_raw_dot, result_gate, result_rstd_x, result_rstd_k = engram_gate_ref(
        hidden_states.to('npu'), k.to('npu'), v.to('npu'),
        weight_hidden.to('npu'), weight_embed.to('npu'),
        CLAMP_VALUE, EPS, save_for_backward=True,
    )
    result_out = result_out.cpu()
    result_raw_dot = result_raw_dot.cpu()
    result_gate = result_gate.cpu()
    result_rstd_x = result_rstd_x.cpu()
    result_rstd_k = result_rstd_k.cpu()

    param_id = make_param_id({
        'num_tokens': NUM_TOKENS,
        'hidden': HIDDEN,
        'hc_mult': HC_MULT,
        'dtype': torch.bfloat16,
    })
    torch.testing.assert_close(result_out, golden_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(result_raw_dot, golden_raw_dot, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(result_gate, golden_gate, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(result_rstd_x, golden_rstd_x, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(result_rstd_k, golden_rstd_k, rtol=1e-2, atol=1e-2)
