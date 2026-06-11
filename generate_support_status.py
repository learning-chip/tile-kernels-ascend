#!/usr/bin/env python3

import csv
import importlib
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


KERNEL_LOC = {
    'engram': {
        'engram_hash': 50,
        'engram_gate_fwd': 140,
        'engram_gate_bwd': 227,
        'grad_w_reduce': 49,
        'fused_weight': 27,
    },
    'mhc': {
        'expand_to_mhc': 47,
        'sinkhorn_normalize': 118,
        'mhc_head_compute_mix': 65,
        'mhc_pre_split_mixes': 124,
        'mhc_pre_apply_mix': 78,
        'mhc_post': 114,
        'mhc_pre_norm_fn': 228,
        'mhc_pre_big_fuse': 96,
        'mhc_multilayer_recompute': 91,
    },
    'moe': {
        'normalize_weight': 29,
        'group_count': 32,
        'reduce_fused': 52,
        'expand_to_fused': 70,
        'expand_to_fused_with_sf': 70,
        'topk_gate': 40,
        'mask_indices_by_tp': 34,
        'inplace_unique_group_indices': 34,
        'top2_sum_gate': 221,
        'aux_fi': 33,
        'get_fused_mapping': 120,
        'topk_sum_and_topk_group_idx': 48,
    },
    'quant': {
        'per_channel_cast_fused': 98,
        'per_token_cast': 113,
        'swiglu_forward_and_per_channel_cast_and_transpose': 139,
        'cast_back': 46,
        'per_channel_cast_and_transpose': 57,
        'swiglu_backward_and_per_token_cast': 120,
        'per_block_cast': 112,
        'per_token_cast_to_e5m6': 69,
        'per_block_cast_lossless': 96,
        'swiglu_forward_and_per_token_cast': 134,
        'cast_back_e5m6': 64,
        'per_channel_cast': 0,
    },
    'transpose': {
        'batched_transpose': 46,
        'transpose': 0,
    },
}


TORCH_FUNC_MAP = {
    ('engram', 'engram_hash'): ('tile_kernels_ascend.torch.engram', 'engram_hash_ref'),
    ('engram', 'engram_gate_fwd'): ('tile_kernels_ascend.torch.engram', 'engram_gate_ref'),
    ('engram', 'engram_gate_bwd'): None,
    ('engram', 'grad_w_reduce'): ('tile_kernels_ascend.torch.engram', 'grad_w_reduce_ref'),
    ('engram', 'fused_weight'): ('tile_kernels_ascend.torch.engram', 'fused_weight_ref'),
    ('mhc', 'expand_to_mhc'): ('tile_kernels_ascend.torch.mhc', 'expand_to_mhc_ref'),
    ('mhc', 'sinkhorn_normalize'): ('tile_kernels_ascend.torch.mhc', 'sinkhorn_normalize_ref'),
    ('mhc', 'mhc_head_compute_mix'): ('tile_kernels_ascend.torch.mhc', 'mhc_head_compute_mix_ref'),
    ('mhc', 'mhc_pre_split_mixes'): ('tile_kernels_ascend.torch.mhc', 'mhc_pre_split_mixes_ref'),
    ('mhc', 'mhc_pre_apply_mix'): ('tile_kernels_ascend.torch.mhc', 'mhc_pre_apply_mix_ref'),
    ('mhc', 'mhc_post'): ('tile_kernels_ascend.torch.mhc', 'mhc_post_ref'),
    ('mhc', 'mhc_pre_norm_fn'): ('tile_kernels_ascend.torch.mhc', 'mhc_pre_norm_fn_ref'),
    ('mhc', 'mhc_pre_big_fuse'): ('tile_kernels_ascend.torch.mhc', 'mhc_pre_big_fuse_ref'),
    ('mhc', 'mhc_multilayer_recompute'): None,
    ('moe', 'aux_fi'): ('tile_kernels_ascend.torch.moe', 'aux_fi'),
    ('moe', 'group_count'): ('tile_kernels_ascend.torch.moe', 'group_count'),
    ('moe', 'mask_indices_by_tp'): ('tile_kernels_ascend.torch.moe', 'mask_indices_by_tp'),
    ('moe', 'normalize_weight'): ('tile_kernels_ascend.torch.moe', 'normalize_weight'),
    ('moe', 'inplace_unique_group_indices'): ('tile_kernels_ascend.torch.moe', 'inplace_unique_group_indices'),
    ('moe', 'get_fused_mapping'): ('tile_kernels_ascend.torch.moe', 'get_fused_mapping'),
    ('moe', 'topk_gate'): ('tile_kernels_ascend.torch.moe', 'topk_gate_ref'),
    ('moe', 'top2_sum_gate'): ('tile_kernels_ascend.torch.moe', 'top2_sum_gate'),
    ('moe', 'topk_sum_and_topk_group_idx'): ('tile_kernels_ascend.torch.moe', 'topk_sum_and_topk_group_idx'),
    ('moe', 'expand_to_fused'): ('tile_kernels_ascend.torch.moe.expand_to_fused', 'expand_to_fused'),
    ('moe', 'expand_to_fused_with_sf'): ('tile_kernels_ascend.torch.moe.expand_to_fused', 'expand_to_fused_with_sf'),
    ('moe', 'reduce_fused'): ('tile_kernels_ascend.torch.moe.reduce_fused', 'reduce_fused'),
    ('quant', 'per_token_cast'): ('tile_kernels_ascend.torch.quant.cast', 'cast'),
    ('quant', 'per_channel_cast'): ('tile_kernels_ascend.torch.quant.cast', 'cast'),
    ('quant', 'per_channel_cast_and_transpose'): None,
    ('quant', 'per_channel_cast_fused'): ('tile_kernels_ascend.torch.quant.per_channel_cast_fused', 'per_channel_cast_fused'),
    ('quant', 'per_block_cast'): ('tile_kernels_ascend.torch.quant.cast', 'cast'),
    ('quant', 'per_block_cast_lossless'): None,
    ('quant', 'cast_back'): ('tile_kernels_ascend.torch.quant.cast', 'cast_back'),
    ('quant', 'cast_back_e5m6'): ('tile_kernels_ascend.torch.quant.cast_e5m6', 'cast_back_from_e5m6'),
    ('quant', 'swiglu_forward_and_per_channel_cast_and_transpose'): None,
    ('quant', 'swiglu_forward_and_per_token_cast'): None,
    ('quant', 'swiglu_backward_and_per_token_cast'): None,
    ('quant', 'per_token_cast_to_e5m6'): ('tile_kernels_ascend.torch.quant.cast_e5m6', 'cast_to_e5m6'),
    ('transpose', 'transpose'): ('tile_kernels_ascend.torch.transpose', 'transpose_ref'),
    ('transpose', 'batched_transpose'): ('tile_kernels_ascend.torch.transpose', 'batched_transpose_ref'),
}


ACLNN_NOTES = {
    ('engram', 'engram_hash'): 'No torch_npu API; custom n-gram hashing',
    ('engram', 'engram_gate_fwd'): 'No torch_npu API; custom gated attention',
    ('engram', 'engram_gate_bwd'): 'No torch_npu API; backward pass unavailable',
    ('engram', 'grad_w_reduce'): 'No torch_npu API; custom gradient reduction',
    ('engram', 'fused_weight'): 'No torch_npu API; custom weight fusion',
    ('mhc', 'sinkhorn_normalize'): 'CANN >= 9.0.0 required (npu_mhc_sinkhorn)',
    ('mhc', 'mhc_post'): 'CANN >= 9.0.0 required (npu_mhc_post)',
    ('mhc', 'mhc_pre_big_fuse'): 'CANN >= 9.0.0 required (npu_mhc_pre)',
    ('moe', 'get_fused_mapping'): 'Interface mismatch with npu_moe_init_routing',
    ('moe', 'expand_to_fused'): 'Interface mismatch with npu_moe_init_routing',
    ('moe', 'expand_to_fused_with_sf'): 'Interface mismatch with npu_moe_init_routing',
    ('moe', 'reduce_fused'): 'Interface mismatch with npu_moe_finalize_routing',
    ('moe', 'topk_sum_and_topk_group_idx'): 'No direct torch_npu API',
    ('quant', 'per_channel_cast_and_transpose'): 'No torch_npu fused kernel available',
    ('quant', 'per_block_cast_lossless'): 'No torch_npu API for lossless block cast',
    ('quant', 'cast_back_e5m6'): 'No torch_npu API for e5m6 format',
    ('quant', 'per_token_cast_to_e5m6'): 'No torch_npu API for e5m6 format',
    ('quant', 'swiglu_forward_and_per_channel_cast_and_transpose'): 'No torch_npu fused kernel available',
    ('quant', 'swiglu_forward_and_per_token_cast'): 'No torch_npu fused kernel available',
    ('quant', 'swiglu_backward_and_per_token_cast'): 'No torch_npu fused kernel available',
}


def _has_not_implemented(func):
    try:
        src = inspect.getsource(func)
        return 'NotImplementedError' in src
    except (TypeError, OSError):
        return False


def _check_torch_eager(family, kernel_name):
    mapping = TORCH_FUNC_MAP.get((family, kernel_name))
    if mapping is None:
        return False
    module_path, attr_name = mapping
    try:
        mod = importlib.import_module(module_path)
        func = getattr(mod, attr_name, None)
        if func is None or not callable(func):
            return False
        return not _has_not_implemented(func)
    except Exception:
        return False


def _check_backend(backend, family, kernel_name):
    module_path = f'tile_kernels_ascend.{backend}.{family}'
    try:
        mod = importlib.import_module(module_path)
    except Exception:
        return False
    for attr_name in (kernel_name, f'{kernel_name}_ref'):
        func = getattr(mod, attr_name, None)
        if func is not None and callable(func):
            return not _has_not_implemented(func)
    return False


def _get_note(family, kernel_name, aclnn_supported):
    if aclnn_supported:
        return ''
    return ACLNN_NOTES.get((family, kernel_name), 'No torch_npu API available')


def scan_tests():
    tests_dir = Path(__file__).parent / 'tests'
    coverage = {}
    if not tests_dir.is_dir():
        return coverage
    for entry in sorted(tests_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith(('_', '.')):
            family = entry.name
            kernels = set()
            for test_file in sorted(entry.glob('test_*.py')):
                kernels.add(test_file.stem[len('test_'):])
            if kernels:
                coverage[family] = kernels
    standalone = sorted(tests_dir.glob('test_*.py'))
    if standalone:
        kernels = set()
        for test_file in standalone:
            kernels.add(test_file.stem[len('test_'):])
        coverage['_root'] = kernels
    return coverage


def generate_rows():
    rows = []
    backends = [
        ('torch-eager', _check_torch_eager),
        ('aclnn', lambda f, k: _check_backend('aclnn', f, k)),
        ('pto', lambda f, k: _check_backend('pto', f, k)),
    ]
    for family, kernel_map in KERNEL_LOC.items():
        for kernel_name, ref_loc in kernel_map.items():
            aclnn_supported = _check_backend('aclnn', family, kernel_name)
            for backend_name, checker in backends:
                supported = checker(family, kernel_name)
                note = ''
                if backend_name == 'aclnn':
                    note = _get_note(family, kernel_name, supported)
                rows.append({
                    'family': family,
                    'kernel_name': kernel_name,
                    'backend': backend_name,
                    'supported': supported,
                    'ref_loc_original_tilelang': ref_loc,
                    'aclnn_supported': aclnn_supported if backend_name == 'aclnn' else '',
                    'speedup_vs_cpu': 'N/A',
                    'notes': note,
                })
    return rows


def main():
    test_coverage = scan_tests()
    rows = generate_rows()

    csv_dir = Path(__file__).parent / 'support_status'
    csv_dir.mkdir(exist_ok=True)
    csv_path = csv_dir / 'support_status.csv'
    fieldnames = [
        'family', 'kernel_name', 'backend', 'supported',
        'ref_loc_original_tilelang', 'aclnn_supported', 'speedup_vs_cpu', 'notes',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written to: {csv_path}\n")

    print("=" * 80)
    print("KERNEL SUPPORT STATUS SUMMARY")
    print("=" * 80)

    if test_coverage:
        print("\nTest coverage (tests/):")
        for fam in sorted(test_coverage):
            kernels = sorted(test_coverage[fam])
            label = fam if fam != '_root' else '(root-level)'
            print(f"  {label}: {len(kernels)} test files")
            for k in kernels:
                marker = " [known]" if fam != '_root' and fam in KERNEL_LOC and k in KERNEL_LOC[fam] else " [extra]"
                print(f"    - {k}{marker}")

    family_stats = {}
    total_supported = 0
    total_rows = 0
    aclnn_total_supported = 0
    aclnn_total_kernels = 0
    for r in rows:
        fam = r['family']
        if fam not in family_stats:
            family_stats[fam] = {
                'supported': 0, 'total': 0,
                'aclnn_supported': 0, 'aclnn_total': 0,
            }
        family_stats[fam]['total'] += 1
        total_rows += 1
        if r['supported']:
            family_stats[fam]['supported'] += 1
            total_supported += 1

    aclnn_rows = [r for r in rows if r['backend'] == 'aclnn']
    for r in aclnn_rows:
        fam = r['family']
        family_stats[fam]['aclnn_total'] += 1
        aclnn_total_kernels += 1
        if r['supported']:
            family_stats[fam]['aclnn_supported'] += 1
            aclnn_total_supported += 1

    print(f"\n{'Family':<12} {'All Backends':<24} {'ACLNN only':<24} {'ACLNN Count'}")
    print("-" * 75)
    for fam in ['engram', 'mhc', 'moe', 'quant', 'transpose']:
        if fam not in family_stats:
            continue
        s = family_stats[fam]['supported']
        t = family_stats[fam]['total']
        pct = f"{100 * s / t:.1f}%" if t else "N/A"
        as_ = family_stats[fam]['aclnn_supported']
        at = family_stats[fam]['aclnn_total']
        apct = f"{100 * as_ / at:.1f}%" if at else "N/A"
        all_lbl = f"{s}/{t} ({pct})"
        aclnn_lbl = f"{as_}/{at} ({apct})"
        print(f"{fam:<12} {all_lbl:<24} {aclnn_lbl:<24} {as_}")

    pct_overall = f"{100 * total_supported / total_rows:.1f}%" if total_rows else "N/A"
    apct_overall = f"{100 * aclnn_total_supported / aclnn_total_kernels:.1f}%" if aclnn_total_kernels else "N/A"
    print("-" * 75)
    all_lbl = f"{total_supported}/{total_rows} ({pct_overall})"
    aclnn_lbl = f"{aclnn_total_supported}/{aclnn_total_kernels} ({apct_overall})"
    print(f"{'OVERALL':<12} {all_lbl:<24} {aclnn_lbl:<24} {aclnn_total_supported}")
    print("=" * 80)

    print("\nDetailed support matrix:")
    hdr = f"{'Family':<12} {'Kernel':<50} {'torch-eager':<14} {'aclnn':<10} {'pto':<10} {'notes'}"
    print(hdr)
    print("-" * 130)
    current_family = None
    for r in rows:
        fam = r['family']
        if fam != current_family:
            if current_family is not None:
                print()
            current_family = fam
        if r['backend'] == 'torch-eager':
            torch_val = r['supported']
            aclnn_val = None
            pto_val = None
            current_note = ''
        elif r['backend'] == 'aclnn':
            aclnn_val = r['supported']
            current_note = r['notes']
        elif r['backend'] == 'pto':
            pto_val = r['supported']
            te = str(torch_val)
            ae = str(aclnn_val)
            pe = str(pto_val)
            print(f"{r['family']:<12} {r['kernel_name']:<50} {te:<14} {ae:<10} {pe:<10} {current_note}")


if __name__ == '__main__':
    main()
