import pytest
import torch

from tile_kernels_ascend.aclnn.transpose import transpose, batched_transpose
from tile_kernels_ascend.torch.transpose import transpose_ref, batched_transpose_ref

from tile_kernels_ascend.aclnn.moe import (
    aux_fi, group_count, mask_indices_by_tp, normalize_weight, inplace_unique_group_indices
)
from tile_kernels_ascend.torch.moe import (
    aux_fi as aux_fi_ref,
    group_count as group_count_ref,
    mask_indices_by_tp as mask_indices_by_tp_ref,
    normalize_weight as normalize_weight_ref,
    inplace_unique_group_indices as inplace_unique_group_indices_ref,
)

from tile_kernels_ascend.aclnn.mhc import (
    expand_to_mhc, mhc_head_compute_mix, mhc_pre_apply_mix, mhc_pre_split_mixes, mhc_pre_norm_fn
)
from tile_kernels_ascend.torch.mhc import (
    expand_to_mhc_ref,
    mhc_head_compute_mix_ref,
    mhc_pre_apply_mix_ref,
    mhc_pre_split_mixes_ref,
    mhc_pre_norm_fn_ref,
)

try:
    from tile_kernels_ascend.aclnn.quant import per_token_cast
    from tile_kernels_ascend.torch.quant.cast import cast as per_token_cast_ref
    QUANT_AVAILABLE = True
except ImportError:
    QUANT_AVAILABLE = False


def _has_npu():
    return hasattr(torch, 'npu') and torch.npu.is_available()


@pytest.mark.skipif(not _has_npu(), reason='NPU not available')
class TestTransposeACLNN:
    def test_transpose_2d(self):
        x = torch.randn(32, 64, dtype=torch.float32)
        ref = transpose_ref(x)
        x_npu = x.npu()
        out = transpose(x_npu).cpu()
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_transpose_3d(self):
        x = torch.randn(4, 32, 64, dtype=torch.float32)
        ref = batched_transpose_ref(x)
        x_npu = x.npu()
        out = batched_transpose(x_npu).cpu()
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _has_npu(), reason='NPU not available')
class TestMoEACLNN:
    def test_aux_fi(self):
        topk_idx = torch.randint(0, 8, (16, 2), dtype=torch.int64)
        ref = aux_fi_ref(topk_idx, 8, 2)
        out = aux_fi(topk_idx.npu(), 8, 2).cpu()
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_group_count(self):
        group_idx = torch.randint(0, 4, (16, 2), dtype=torch.int64)
        ref = group_count_ref(group_idx, 4)
        out = group_count(group_idx.npu(), 4).cpu()
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_mask_indices_by_tp(self):
        indices = torch.randint(0, 32, (16, 2), dtype=torch.int64)
        ref = mask_indices_by_tp_ref(indices, 32, 1, 0, 1)
        out = mask_indices_by_tp(indices.npu(), 32, 1, 0, 1).cpu()
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_normalize_weight(self):
        topk_weights = torch.randn(16, 2, dtype=torch.float32).abs()
        denom_ref, norm_ref = normalize_weight_ref(topk_weights)
        denom_out, norm_out = normalize_weight(topk_weights.npu())
        torch.testing.assert_close(denom_out.cpu(), denom_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(norm_out.cpu(), norm_ref, rtol=1e-5, atol=1e-5)

    def test_inplace_unique_group_indices(self):
        group_indices = torch.randint(0, 4, (16, 4), dtype=torch.int64)
        group_indices_ref_data = group_indices.clone()
        inplace_unique_group_indices_ref(group_indices_ref_data, 4)

        group_indices_npu = group_indices.npu()
        inplace_unique_group_indices(group_indices_npu, 4)
        torch.testing.assert_close(group_indices_npu.cpu(), group_indices_ref_data, rtol=0, atol=0)


@pytest.mark.skipif(not _has_npu(), reason='NPU not available')
class TestMHCACLNN:
    def test_expand_to_mhc(self):
        hidden = torch.randn(4, 8, 16, dtype=torch.float32)
        mhc_mult = 2
        ref = expand_to_mhc_ref(hidden, mhc_mult)
        out = expand_to_mhc(hidden.npu(), mhc_mult).cpu()
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_mhc_head_compute_mix(self):
        input_mix = torch.randn(4, 8, dtype=torch.float32)
        mhc_scale = torch.randn(8, dtype=torch.float32)
        mhc_base = torch.randn(8, dtype=torch.float32)
        mhc_pre_eps = 1e-6
        ref = mhc_head_compute_mix_ref(input_mix, mhc_scale, mhc_base, mhc_pre_eps)
        out = mhc_head_compute_mix(input_mix.npu(), mhc_scale.npu(), mhc_base.npu(), mhc_pre_eps).cpu()
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_mhc_pre_split_mixes(self):
        input_mixes = torch.randn(4, 8, (2 + 2 + 4), dtype=torch.float32)
        mhc_scale = torch.randn(3, dtype=torch.float32)
        mhc_base = torch.randn((2 + 2 + 4), dtype=torch.float32)
        mhc_mult = 2
        mhc_post_mult_value = 2.0
        mhc_pre_eps = 1e-6
        pre_ref, post_ref, comb_ref = mhc_pre_split_mixes_ref(
            input_mixes, mhc_scale, mhc_base, mhc_mult, mhc_post_mult_value, mhc_pre_eps
        )
        pre_out, post_out, comb_out = mhc_pre_split_mixes(
            input_mixes.npu(), mhc_scale.npu(), mhc_base.npu(), mhc_mult, mhc_post_mult_value, mhc_pre_eps
        )
        torch.testing.assert_close(pre_out.cpu(), pre_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(post_out.cpu(), post_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(comb_out.cpu(), comb_ref, rtol=1e-5, atol=1e-5)

    def test_mhc_pre_apply_mix(self):
        x = torch.randn(4, 8, 2, 16, dtype=torch.float32)
        mix = torch.randn(4, 8, 2, 1, dtype=torch.float32)
        ref = mhc_pre_apply_mix_ref(x, mix)
        out = mhc_pre_apply_mix(x.npu(), mix.npu()).cpu()
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_mhc_pre_norm_fn(self):
        residual = torch.randn(2, 4, 1, 8, dtype=torch.float32)
        mhc_mult = 2
        rms_group_size = 8
        mhc_fn = torch.randn(mhc_mult, rms_group_size, dtype=torch.float32)
        mhc_norm_weight = torch.randn(mhc_mult, rms_group_size, dtype=torch.float32)
        mhc_norm_eps = 1e-6
        ref = mhc_pre_norm_fn_ref(residual, mhc_fn, mhc_norm_weight, mhc_norm_eps)
        out = mhc_pre_norm_fn(residual.npu(), mhc_fn.npu(), mhc_norm_weight.npu(), mhc_norm_eps).cpu()
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _has_npu(), reason='NPU not available')
@pytest.mark.skipif(not QUANT_AVAILABLE, reason='Quant module not available')
class TestQuantACLNN:
    def test_per_token_cast(self):
        x = torch.randn(16, 64, dtype=torch.bfloat16).npu()
        quant_out, sf_out = per_token_cast(x, fmt='e4m3')
        assert quant_out.shape == x.shape
        assert quant_out.dtype in [torch.float8_e4m3fn, torch.int8]
        assert sf_out.shape[0] == x.shape[0]
