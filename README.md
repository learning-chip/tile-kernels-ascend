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

| Family | Kernel | Reason |
|--------|--------|--------|
| engram | all | No torch_npu APIs; custom architecture |
| moe | `get_fused_mapping` | Interface mismatch with `npu_moe_init_routing` |
| moe | `expand_to_fused` | Interface mismatch with `npu_moe_init_routing` |
| moe | `expand_to_fused_with_sf` | Not yet implemented in torch_npu |
| moe | `reduce_fused` | Interface mismatch with `npu_moe_finalize_routing` |
| moe | `topk_sum_and_topk_group_idx` | No direct torch_npu API |
| mhc | `sinkhorn_normalize` | CANN >= 9.0.0 required (`npu_mhc_sinkhorn`) |
| mhc | `mhc_post` | CANN >= 9.0.0 required (`npu_mhc_post`) |
| mhc | `mhc_pre_big_fuse` | CANN >= 9.0.0 required (`npu_mhc_pre`) |
| quant | `per_channel_cast_and_transpose` | No torch_npu fused kernel available |
| quant | `per_block_cast_lossless` | No torch_npu API for lossless block cast |
| quant | `cast_back_e5m6` | No torch_npu API for e5m6 format |
| quant | `per_token_cast_to_e5m6` | No torch_npu API for e5m6 format |
| quant | `swiglu_backward_and_per_token_cast` | No SwiGLU backward API in torch_npu |

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

| Family | # Kernels | Original @jit LoC |
|--------|-----------|-------------------|
| engram | 5 | 493 |
| mhc | 20 | 961 |
| moe | 11 | 713 |
| quant | 11 | 1,048 |
| transpose | 1 | 46 |
| **Total** | **48** | **3,261** |

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
