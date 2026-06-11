import pytest
import torch
from tile_kernels_ascend.torch.mhc import mhc_pre_split_mixes_ref

npu_available = hasattr(torch, "npu") and torch.npu.is_available()
device = "npu" if npu_available else "cpu"

MHC_MULT = 4
MIX_SIZE = 2 * MHC_MULT + MHC_MULT * MHC_MULT


def _make_inputs(n0, n1, mix_size, mhc_mult):
    input_mixes = torch.randn(n0, n1, mix_size, dtype=torch.float32)
    mhc_scale = torch.randn(3, dtype=torch.float32)
    mhc_base = torch.randn(mix_size, dtype=torch.float32)
    return input_mixes, mhc_scale, mhc_base


class TestPreSplitMixes:
    n0, n1, mhc_mult = 1, 256, 4

    def test_forward(self):
        input_mixes, mhc_scale, mhc_base = _make_inputs(self.n0, self.n1, MIX_SIZE, self.mhc_mult)
        mult_val, eps = 2.0, 1e-6
        with torch.no_grad():
            golden = mhc_pre_split_mixes_ref(input_mixes.clone(), mhc_scale, mhc_base, self.mhc_mult, mult_val, eps)
            im_npu = input_mixes.clone().to(device)
            sc_npu = mhc_scale.to(device)
            bs_npu = mhc_base.to(device)
            npu_out = mhc_pre_split_mixes_ref(im_npu, sc_npu, bs_npu, self.mhc_mult, mult_val, eps)
        for g, n in zip(golden, npu_out):
            torch.testing.assert_close(n.cpu(), g, rtol=1e-2, atol=1e-2)

    def test_backward(self):
        input_mixes_base, mhc_scale_base, mhc_base_base = _make_inputs(self.n0, self.n1, MIX_SIZE, self.mhc_mult)
        mult_val, eps = 2.0, 1e-6

        def run(im, sc, bs):
            pre, post, comb = mhc_pre_split_mixes_ref(im, sc, bs, self.mhc_mult, mult_val, eps)
            return pre.float().sum() + post.float().sum() + comb.float().sum()

        im_ref = input_mixes_base.clone().requires_grad_(True)
        sc_ref = mhc_scale_base.clone().requires_grad_(True)
        bs_ref = mhc_base_base.clone().requires_grad_(True)
        run(im_ref, sc_ref, bs_ref).backward()

        im_npu = input_mixes_base.clone().to(device).requires_grad_(True)
        sc_npu = mhc_scale_base.clone().to(device).requires_grad_(True)
        bs_npu = mhc_base_base.clone().to(device).requires_grad_(True)
        run(im_npu, sc_npu, bs_npu).backward()

        torch.testing.assert_close(im_npu.grad.cpu(), im_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(sc_npu.grad.cpu(), sc_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(bs_npu.grad.cpu(), bs_ref.grad, rtol=1e-2, atol=1e-2)
