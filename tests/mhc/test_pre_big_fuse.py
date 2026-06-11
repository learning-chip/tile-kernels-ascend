import pytest
import torch
from tile_kernels_ascend.torch.mhc import mhc_pre_big_fuse_ref

npu_available = hasattr(torch, "npu") and torch.npu.is_available()
device = "npu" if npu_available else "cpu"


def _make_inputs(B, S, n, D):
    nD = n * D
    out_dim = 2 * n + n * n
    x = torch.randn(B, S, n, D, dtype=torch.float32)
    phi = torch.randn(out_dim, nD, dtype=torch.float32) * 0.1
    a_pre = torch.randn(n, dtype=torch.float32)
    a_post = torch.randn(n, dtype=torch.float32)
    a_res = torch.randn(n, n, dtype=torch.float32)
    alpha = (a_pre, a_post, a_res)
    bias = torch.randn(out_dim, dtype=torch.float32) * 0.1
    gamma = torch.randn(nD, dtype=torch.float32) * 0.1
    return x, phi, alpha, bias, gamma


class TestPreBigFuse:
    B, S, n, D = 1, 64, 4, 128

    def test_forward_with_gamma(self):
        x, phi, alpha, bias, gamma = _make_inputs(self.B, self.S, self.n, self.D)
        norm_eps, hc_eps = 1e-6, 1e-6
        with torch.no_grad():
            golden = mhc_pre_big_fuse_ref(x.clone(), phi.clone(), alpha, bias.clone(), gamma.clone(), norm_eps, hc_eps)
            x_npu = x.clone().to(device)
            phi_npu = phi.clone().to(device)
            bias_npu = bias.clone().to(device)
            gamma_npu = gamma.clone().to(device)
            alpha_npu = tuple(a.clone().to(device) for a in alpha)
            npu_out = mhc_pre_big_fuse_ref(x_npu, phi_npu, alpha_npu, bias_npu, gamma_npu, norm_eps, hc_eps)
        for g, n in zip(golden, npu_out):
            torch.testing.assert_close(n.cpu(), g, rtol=1e-2, atol=1e-2)

    def test_forward_without_gamma(self):
        x, phi, alpha, bias, _ = _make_inputs(self.B, self.S, self.n, self.D)
        norm_eps, hc_eps = 1e-6, 1e-6
        with torch.no_grad():
            golden = mhc_pre_big_fuse_ref(x.clone(), phi.clone(), alpha, bias.clone(), None, norm_eps, hc_eps)
            x_npu = x.clone().to(device)
            phi_npu = phi.clone().to(device)
            bias_npu = bias.clone().to(device)
            alpha_npu = tuple(a.clone().to(device) for a in alpha)
            npu_out = mhc_pre_big_fuse_ref(x_npu, phi_npu, alpha_npu, bias_npu, None, norm_eps, hc_eps)
        for g, n in zip(golden, npu_out):
            torch.testing.assert_close(n.cpu(), g, rtol=1e-2, atol=1e-2)
