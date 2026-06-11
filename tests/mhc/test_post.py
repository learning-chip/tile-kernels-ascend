import pytest
import torch
from tile_kernels_ascend.torch.mhc import mhc_post_ref

npu_available = hasattr(torch, "npu") and torch.npu.is_available()
device = "npu" if npu_available else "cpu"


def _make_inputs(n0, n1, h, mhc_mult):
    x = torch.randn(n0, n1, h, dtype=torch.bfloat16)
    residual = torch.randn(n0, n1, mhc_mult, h, dtype=torch.bfloat16)
    post_layer_mix = torch.rand(n0, n1, mhc_mult, 1, dtype=torch.float32)
    comb_res_mix = torch.randn(n0, n1, mhc_mult, mhc_mult, dtype=torch.float32)
    return x, residual, post_layer_mix, comb_res_mix


class TestPost:
    n0, n1, h, mhc_mult = 1, 256, 128, 4

    def test_forward(self):
        x, residual, post_layer_mix, comb_res_mix = _make_inputs(self.n0, self.n1, self.h, self.mhc_mult)
        with torch.no_grad():
            golden = mhc_post_ref(x.clone(), residual.clone(), post_layer_mix.clone(), comb_res_mix.clone())
            x_npu = x.clone().to(device)
            r_npu = residual.clone().to(device)
            p_npu = post_layer_mix.clone().to(device)
            c_npu = comb_res_mix.clone().to(device)
            npu_out = mhc_post_ref(x_npu, r_npu, p_npu, c_npu)
        torch.testing.assert_close(npu_out.cpu(), golden, rtol=1e-2, atol=1e-2)

    def test_backward(self):
        x_base, res_base, post_base, comb_base = _make_inputs(self.n0, self.n1, self.h, self.mhc_mult)

        x_ref = x_base.float().clone().requires_grad_(True)
        r_ref = res_base.float().clone().requires_grad_(True)
        p_ref = post_base.clone().requires_grad_(True)
        c_ref = comb_base.clone().requires_grad_(True)
        out_ref = mhc_post_ref(x_ref, r_ref, p_ref, c_ref)
        out_ref.float().sum().backward()

        x_npu = x_base.float().clone().to(device).requires_grad_(True)
        r_npu = res_base.float().clone().to(device).requires_grad_(True)
        p_npu = post_base.clone().to(device).requires_grad_(True)
        c_npu = comb_base.clone().to(device).requires_grad_(True)
        out_npu = mhc_post_ref(x_npu, r_npu, p_npu, c_npu)
        out_npu.float().sum().backward()

        torch.testing.assert_close(x_npu.grad.cpu(), x_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(r_npu.grad.cpu(), r_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(p_npu.grad.cpu(), p_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(c_npu.grad.cpu(), c_ref.grad, rtol=1e-2, atol=1e-2)
