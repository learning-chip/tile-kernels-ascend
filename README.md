# tile-kernels-ascend

Port of [TileKernels](../TileKernels) custom kernels to Ascend NPU.
Each original tilelang kernel is represented as a torch-eager reference, with ACLNN and PTO fused-kernel backends as future targets.

## Structure

```
tile_kernels_ascend/
  torch/        # PyTorch eager implementations (reference code)
  aclnn/        # ACLNN backend (torch_npu fused APIs)
  pto/          # PTO backend  (placeholder)
tests/          # pytest accuracy: NPU eager vs CPU reference
benchmark/      # pytest benchmarks with @pytest.mark.benchmark flag
support_status/ # auto-generated CSV of support coverage
```

## Test Suite

```bash
# Default tests (fast subset) — finishes in ~10 seconds
pytest tests/ -q

# ACLNN backend correctness tests
pytest tests/test_aclnn_backends.py -v

# Full benchmark run (requires --run-benchmark)
pytest benchmark/ -q --run-benchmark --benchmark-output=benchmark_results.jsonl
```

Expected test time: ~10 seconds for default tests.

## ACLNN Backend Status

The ACLNN backend wraps `torch_npu` fused APIs for Ascend NPU acceleration.

### Supported Kernels

| Family | Kernel | Notes |
|--------|--------|-------|
| transpose | `transpose` | NPU native transpose |
| transpose | `batched_transpose` | Batched dim-swap |
| moe | `aux_fi` | Auxiliary load balancing |
| moe | `group_count` | Expert group counting |
| moe | `mask_indices_by_tp` | TP-aware index masking |
| moe | `normalize_weight` | Top-k weight normalization |
| moe | `inplace_unique_group_indices` | De-duplicate group indices |
| moe | `topk_gate` | `torch_npu.npu_moe_gating_top_k_softmax` / `npu_moe_gating_top_k` |
| moe | `top2_sum_gate` | `torch_npu.npu_moe_gating_top_k` (group routing) |
| moe | `get_fused_mapping` | `npu_moe_init_routing` + sort-based post-processing |
| moe | `expand_to_fused` | `npu_moe_init_routing` (expansion with position remap) |
| moe | `expand_to_fused_with_sf` | `npu_moe_init_routing` × 2 calls (data + sf) |
| moe | `reduce_fused` | `npu_moe_finalize_routing` (column-major reshape of `token_topk_to_pos`) |
| moe | `topk_sum_and_topk_group_idx` | `torch.topk` + `torch.sort` (NPU-native; `npu_moe_gating_top_k` doesn't expose group indices) |
| mhc | `expand_to_mhc` | Expand hidden for MHC heads |
| mhc | `mhc_head_compute_mix` | Per-head mix coefficients |
| mhc | `mhc_pre_split_mixes` | Split pre/post/comb mixes |
| mhc | `mhc_pre_apply_mix` | Apply pre-mix to residual |
| mhc | `mhc_pre_norm_fn` | Normalized RMS mixing |
| mhc | `mhc_multilayer_recompute` | Multi-layer recompute loop |
| quant | `per_token_cast` | `torch_npu.npu_dynamic_quant` |
| quant | `per_channel_cast` | `torch_npu.npu_dynamic_quant` |
| quant | `per_channel_cast_fused` | `torch_npu.npu_dynamic_quant` |
| quant | `per_block_cast` | `torch_npu.npu_dynamic_block_quant` |
| quant | `cast_back` | `torch_npu.npu_anti_quant` |
| quant | `swiglu_forward_and_per_channel_cast_and_transpose` | `npu_swiglu_quant` (int8 output, not FP8 e4m3) |
| quant | `swiglu_forward_and_per_token_cast` | `npu_swiglu_quant` per-token (int8 output, not FP8 e4m3) |

### Unsupported Kernels and Reasons

| Family | Kernel | Reason | Matchability |
|--------|--------|--------|--------------|
| engram | `engram_hash` | No torch_npu API; custom n-gram hashing | too different |
| engram | `engram_gate_fwd` | No torch_npu API; custom gated attention | too different |
| engram | `engram_gate_bwd` | No torch_npu API; backward pass unavailable | too different |
| engram | `grad_w_reduce` | No torch_npu API; custom gradient reduction | too different |
| engram | `fused_weight` | No torch_npu API; custom weight fusion | too different |
| mhc | `sinkhorn_normalize` | CANN >= 9.0.0 required (`npu_mhc_sinkhorn` — API exists, runtime unavailable on this cluster) | can match with a bit pre/post processing (on CANN ≥ 9.0) |
| mhc | `mhc_post` | CANN >= 9.0.0 required (`npu_mhc_post` — API exists, runtime unavailable) | can match with a bit pre/post processing (on CANN ≥ 9.0) |
| mhc | `mhc_pre_big_fuse` | CANN >= 9.0.0 required (`npu_mhc_pre` — API exists, runtime unavailable) | can match with a bit pre/post processing (on CANN ≥ 9.0) |
| quant | `per_channel_cast_and_transpose` | No torch_npu fused cast+transpose | too different |
| quant | `per_block_cast_lossless` | No torch_npu API for lossless block cast | too different |
| quant | `cast_back_e5m6` | No torch_npu API for e5m6 format | too different |
| quant | `per_token_cast_to_e5m6` | No torch_npu API for e5m6 format | too different |
| quant | `swiglu_backward_and_per_token_cast` | No SwiGLU backward API in torch_npu | too different |

## Benchmarks

*(Reserved for future work.)*

Run benchmarks with:
```bash
pytest benchmark/ -q --run-benchmark --benchmark-output=benchmark_results.jsonl
```

## Generating Support Report

```bash
python generate_support_status.py
# Writes support_status/support_status.csv and prints a summary
```

## Reference LoC (Original tilelang)

Per-kernel LoC of the kernel body under `@tilelang.jit` in the original TileKernels repo (excluding testing code, blank lines, comments). See [`reference_loc_detailed.md`](reference_loc_detailed.md) for the full breakdown with source file paths.

| Kernel | Family | LoC |
|--------|--------|----:|
| `engram_grad_w_reduce` | engram | 49 |
| `engram_hash` | engram | 50 |
| `engram_gate_fwd` | engram | 140 |
| `engram_gate_bwd` | engram | 227 |
| `fused_weight` | engram | 27 |
| **engram subtotal** | | **493** |
| `mhc_multilayer_recompute` | mhc | 91 |
| `mhc_pre_big_fuse` | mhc | 96 |
| `mhc_pre_norm_fn` | mhc | 228 |
| `mhc_pre_apply_mix` | mhc | 78 |
| `mhc_pre_split_mixes` | mhc | 124 |
| `mhc_post` | mhc | 114 |
| `mhc_head_compute_mix` | mhc | 65 |
| `mhc_sinkhorn_normalize` | mhc | 118 |
| `mhc_expand_to_mhc` | mhc | 47 |
| **mhc subtotal** | | **961** |
| `moe_top2_sum_gate` | moe | 221 |
| `moe_get_fused_mapping` | moe | 120 |
| `moe_expand_to_fused` | moe | 70 |
| `moe_reduce_fused` | moe | 52 |
| `moe_topk_sum_and_topk_idx` | moe | 48 |
| `moe_topk_gate` | moe | 40 |
| `moe_mask_indices_by_tp` | moe | 34 |
| `moe_inplace_unique` | moe | 34 |
| `moe_group_count` | moe | 32 |
| `moe_aux_fi` | moe | 33 |
| `moe_normalize_weight` | moe | 29 |
| **moe subtotal** | | **713** |
| `quant_swiglu_fwd_per_channel` | quant | 139 |
| `quant_swiglu_fwd_per_token` | quant | 134 |
| `quant_swiglu_bwd_per_token` | quant | 120 |
| `quant_per_token_cast` | quant | 113 |
| `quant_per_block_cast` | quant | 112 |
| `quant_per_channel_cast_fused` | quant | 98 |
| `quant_per_block_cast_lossless` | quant | 96 |
| `quant_per_channel_cast_and_transpose` | quant | 57 |
| `quant_per_token_cast_to_e5m6` | quant | 69 |
| `quant_cast_back` | quant | 46 |
| `quant_cast_back_e5m6` | quant | 64 |
| **quant subtotal** | | **1,048** |
| `transpose_batched` | transpose | 46 |
| **transpose subtotal** | | **46** |
| | | |
| **GRAND TOTAL** | | **3,261** |

## Environment

- PyTorch + torch_npu 2.9.0
- NPU: Ascend 910B2 x 8
- CANN version: 25.5.1 (V100R001C23SPC006B220)
- CPU golden reference: PyTorch eager on CPU
- NPU mock kernel: same PyTorch eager code on NPU

Note: `mhc_pre_big_fuse`, `mhc_post`, and `sinkhorn_normalize` require CANN >= 9.0.0 for the fused `npu_mhc_pre` / `npu_mhc_post` / `npu_mhc_sinkhorn` APIs.

### Reproducing CANN Version Error

On CANN < 9.0.0 (e.g. 25.5.1 / V100R001C23SPC006B220), the MHC fused APIs are unavailable. To reproduce:

```bash
# npu_mhc_pre
python -c "
import torch, torch_npu
T, n, D = 128, 4, 512
x = torch.randn(T, n, D, dtype=torch.bfloat16).npu()
phi = torch.randn(n*n+2*n, n*D, dtype=torch.float32).npu()
alpha = torch.tensor([1.,1.,1.], dtype=torch.float32).npu()
bias = torch.zeros(n*n+2*n, dtype=torch.float32).npu()
torch_npu.npu_mhc_pre(x, phi, alpha, bias)
"
# Error: torch_npu.npu_mhc_pre requires CANN >= 9.0.0 and aclnnMhcPre support. Please upgrade CANN.

# npu_mhc_post
python -c "
import torch, torch_npu
x = torch.randn(1, 128, 4, dtype=torch.bfloat16).npu()
h_res = torch.randn(1, 128, 4, 4, dtype=torch.float32).npu()
h_out = torch.randn(1, 128, 512, dtype=torch.bfloat16).npu()
h_post = torch.randn(1, 128, 4, dtype=torch.float32).npu()
torch_npu.npu_mhc_post(x, h_res, h_out, h_post)
"
# Error: aclnnMhcPost or aclnnMhcPostGetWorkspaceSize not in libopapi.so, or libopapi.so not found.

# npu_mhc_sinkhorn
python -c "
import torch, torch_npu
x = torch.randn(1, 128, 4, 4, dtype=torch.float32).npu()
torch_npu.npu_mhc_sinkhorn(x)
"
# Error: aclnnMhcSinkhorn or aclnnMhcSinkhornGetWorkspaceSize not in libopapi.so, or libopapi.so not found.
```
