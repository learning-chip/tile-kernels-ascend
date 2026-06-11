import pytest
import torch
from tile_kernels_ascend.torch.mhc import mhc_pre_norm_fn_ref

npu_available = hasattr(torch, "npu") and torch.npu.is_available()
device = "npu" if npu_available else "cpu"

MHC_MULT = 4
N1 = 256
HIDDEN = 128


def _make_inputs(with_weight=True):
    residual = torch.randn(1, N1, 1, HIDDEN, dtype=torch.float32)
    mhc_fn = torch.randn(MHC_MULT, 1, HIDDEN, dtype=torch.float32)
    mhc_norm_weight = torch.randn(MHC_MULT, 1, HIDDEN, dtype=torch.float32) if with_weight else None
    eps = 1e-6
    return residual, mhc_fn, mhc_norm_weight, eps


class TestNormFnWithWeight:
    def test_forward(self):
        residual, mhc_fn, weight, eps = _make_inputs(with_weight=True)
        with torch.no_grad():
            golden = mhc_pre_norm_fn_ref(residual.clone(), mhc_fn.clone(), weight.clone(), eps)
            r_npu = residual.clone().to(device)
            f_npu = mhc_fn.clone().to(device)
            w_npu = weight.clone().to(device)
            npu_out = mhc_pre_norm_fn_ref(r_npu, f_npu, w_npu, eps)
        torch.testing.assert_close(npu_out.cpu(), golden, rtol=1e-2, atol=1e-2)

    def test_backward(self):
        residual_base, mhc_fn_base, weight_base, eps = _make_inputs(with_weight=True)

        r_ref = residual_base.clone().requires_grad_(True)
        f_ref = mhc_fn_base.clone().requires_grad_(True)
        w_ref = weight_base.clone().requires_grad_(True)
        out_ref = mhc_pre_norm_fn_ref(r_ref, f_ref, w_ref, eps)
        out_ref.float().sum().backward()

        r_npu = residual_base.clone().to(device).requires_grad_(True)
        f_npu = mhc_fn_base.clone().to(device).requires_grad_(True)
        w_npu = weight_base.clone().to(device).requires_grad_(True)
        out_npu = mhc_pre_norm_fn_ref(r_npu, f_npu, w_npu, eps)
        out_npu.float().sum().backward()

        torch.testing.assert_close(r_npu.grad.cpu(), r_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(f_npu.grad.cpu(), f_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(w_npu.grad.cpu(), w_ref.grad, rtol=1e-2, atol=1e-2)


class TestNormFnWithoutWeight:
    def test_forward(self):
        residual, mhc_fn, _, eps = _make_inputs(with_weight=False)
        with torch.no_grad():
            golden = mhc_pre_norm_fn_ref(residual.clone(), mhc_fn.clone(), None, eps)
            r_npu = residual.clone().to(device)
            f_npu = mhc_fn.clone().to(device)
            npu_out = mhc_pre_norm_fn_ref(r_npu, f_npu, None, eps)
        torch.testing.assert_close(npu_out.cpu(), golden, rtol=1e-2, atol=1e-2)

    def test_backward(self):
        residual_base, mhc_fn_base, _, eps = _make_inputs(with_weight=False)

        r_ref = residual_base.clone().requires_grad_(True)
        f_ref = mhc_fn_base.clone().requires_grad_(True)
        out_ref = mhc_pre_norm_fn_ref(r_ref, f_ref, None, eps)
        out_ref.float().sum().backward()

        r_npu = residual_base.clone().to(device).requires_grad_(True)
        f_npu = mhc_fn_base.clone().to(device).requires_grad_(True)
        out_npu = mhc_pre_norm_fn_ref(r_npu, f_npu, None, eps)
        out_npu.float().sum().backward()

        torch.testing.assert_close(r_npu.grad.cpu(), r_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(f_npu.grad.cpu(), f_ref.grad, rtol=1e-2, atol=1e-2)
