import pytest
import torch

from tile_kernels_ascend.torch.engram import engram_gate_ref
from tile_kernels_ascend.torch.utils import make_param_id


NUM_TOKENS = 256
HC_MULT = 4
HIDDEN = 512
CLAMP_VALUE = 0.01
EPS = 1e-6


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_engram_gate_bwd():
    torch.manual_seed(42)
    hidden_states_data = torch.randn(NUM_TOKENS, HC_MULT, HIDDEN)
    k_data = torch.randn(NUM_TOKENS, HC_MULT, HIDDEN)
    v_data = torch.randn(NUM_TOKENS, HIDDEN)
    weight_hidden_data = torch.randn(HC_MULT, HIDDEN)
    weight_embed_data = torch.randn(HC_MULT, HIDDEN)

    def _run(device):
        hs = hidden_states_data.clone().to(device).requires_grad_(True)
        k = k_data.clone().to(device).requires_grad_(True)
        v = v_data.clone().to(device).requires_grad_(True)
        wh = weight_hidden_data.clone().to(device).requires_grad_(True)
        we = weight_embed_data.clone().to(device).requires_grad_(True)
        output, raw_dot, gate_score, rstd_x, rstd_k = engram_gate_ref(
            hs, k, v, wh, we, CLAMP_VALUE, EPS, save_for_backward=True,
        )
        grad_output = torch.ones_like(output)
        output.backward(grad_output)
        return hs.grad.clone().cpu(), k.grad.clone().cpu(), v.grad.clone().cpu(), wh.grad.clone().cpu(), we.grad.clone().cpu()

    golden_grads = _run('cpu')
    result_grads = _run('npu')

    param_id = make_param_id({
        'num_tokens': NUM_TOKENS,
        'hidden': HIDDEN,
        'hc_mult': HC_MULT,
    })

    for g_golden, g_result, name in zip(
        golden_grads, result_grads,
        ['hidden_states', 'k', 'v', 'weight_hidden', 'weight_embed'],
    ):
        torch.testing.assert_close(g_result, g_golden, rtol=1e-2, atol=1e-2)
