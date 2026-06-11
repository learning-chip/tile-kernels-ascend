import pytest
import torch
from tile_kernels_ascend.torch.mhc import mhc_pre_apply_mix_ref, mhc_post_ref

npu_available = hasattr(torch, "npu") and torch.npu.is_available()
device = "npu" if npu_available else "cpu"

BS, SEQ, MHC, HIDDEN = 1, 64, 4, 128
NUM_LAYERS = 3
NUM_POST = 3


def _multilayer_forward(residual, pre_mixes, post_mixes, comb_mixes):
    for mix, post_mix, comb in zip(pre_mixes, post_mixes, comb_mixes):
        x = mhc_pre_apply_mix_ref(residual, mix)
        residual = mhc_post_ref(x, residual, post_mix, comb)
    return residual


def _make_inputs(num_layers):
    residual = torch.randn(BS, SEQ, MHC, HIDDEN, dtype=torch.float32)
    pre_mixes = [torch.softmax(torch.randn(BS, SEQ, MHC, HIDDEN, dtype=torch.float32), dim=-2)
                 for _ in range(num_layers)]
    post_mixes = [torch.rand(BS, SEQ, MHC, 1, dtype=torch.float32) for _ in range(num_layers)]
    comb_mixes = [torch.randn(BS, SEQ, MHC, MHC, dtype=torch.float32) * 0.1 for _ in range(num_layers)]
    return residual, pre_mixes, post_mixes, comb_mixes


class TestMultilayerRecompute:
    def test_forward(self):
        residual, pre_mixes, post_mixes, comb_mixes = _make_inputs(NUM_LAYERS)
        with torch.no_grad():
            golden = _multilayer_forward(residual.clone(), pre_mixes, post_mixes, comb_mixes)
            r_npu = residual.clone().to(device)
            pm_npu = [m.clone().to(device) for m in pre_mixes]
            psm_npu = [m.clone().to(device) for m in post_mixes]
            cm_npu = [m.clone().to(device) for m in comb_mixes]
            npu_out = _multilayer_forward(r_npu, pm_npu, psm_npu, cm_npu)
        torch.testing.assert_close(npu_out.cpu(), golden, rtol=1e-2, atol=1e-2)

    def test_forward_num_post(self):
        residual, pre_mixes, post_mixes, comb_mixes = _make_inputs(NUM_POST)
        with torch.no_grad():
            golden = _multilayer_forward(residual.clone(), pre_mixes, post_mixes, comb_mixes)
            r_npu = residual.clone().to(device)
            pm_npu = [m.clone().to(device) for m in pre_mixes]
            psm_npu = [m.clone().to(device) for m in post_mixes]
            cm_npu = [m.clone().to(device) for m in comb_mixes]
            npu_out = _multilayer_forward(r_npu, pm_npu, psm_npu, cm_npu)
        torch.testing.assert_close(npu_out.cpu(), golden, rtol=1e-2, atol=1e-2)
        assert npu_out.shape == (BS, SEQ, MHC, HIDDEN)
