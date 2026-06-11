import pytest
import torch
from tile_kernels_ascend.torch.mhc import mhc_head_compute_mix_ref

npu_available = hasattr(torch, "npu") and torch.npu.is_available()
device = "npu" if npu_available else "cpu"

MHC_MULT = 4
MIX_SIZE = 2 * MHC_MULT + MHC_MULT * MHC_MULT


def _make_inputs(n0, n1, mix_size):
    input_mix = torch.randn(n0, n1, mix_size, dtype=torch.float32)
    mhc_scale = torch.randn(mix_size, dtype=torch.float32)
    mhc_base = torch.randn(mix_size, dtype=torch.float32)
    return input_mix, mhc_scale, mhc_base


class TestHeadComputeMix:
    n0, n1 = 1, 256

    def test_forward(self):
        input_mix, mhc_scale, mhc_base = _make_inputs(self.n0, self.n1, MIX_SIZE)
        eps = 1e-6
        with torch.no_grad():
            golden = mhc_head_compute_mix_ref(input_mix.clone(), mhc_scale, mhc_base, eps)
            im_npu = input_mix.clone().to(device)
            sc_npu = mhc_scale.to(device)
            bs_npu = mhc_base.to(device)
            npu_out = mhc_head_compute_mix_ref(im_npu, sc_npu, bs_npu, eps)
        torch.testing.assert_close(npu_out.cpu(), golden, rtol=1e-2, atol=1e-2)

    def test_backward(self):
        input_mix_base, mhc_scale_base, mhc_base_base = _make_inputs(self.n0, self.n1, MIX_SIZE)
        eps = 1e-6

        im_ref = input_mix_base.clone().requires_grad_(True)
        sc_ref = mhc_scale_base.clone().requires_grad_(True)
        bs_ref = mhc_base_base.clone().requires_grad_(True)
        out_ref = mhc_head_compute_mix_ref(im_ref, sc_ref, bs_ref, eps)
        out_ref.sum().backward()

        im_npu = input_mix_base.clone().to(device).requires_grad_(True)
        sc_npu = mhc_scale_base.clone().to(device).requires_grad_(True)
        bs_npu = mhc_base_base.clone().to(device).requires_grad_(True)
        out_npu = mhc_head_compute_mix_ref(im_npu, sc_npu, bs_npu, eps)
        out_npu.sum().backward()

        torch.testing.assert_close(im_npu.grad.cpu(), im_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(sc_npu.grad.cpu(), sc_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(bs_npu.grad.cpu(), bs_ref.grad, rtol=1e-2, atol=1e-2)
