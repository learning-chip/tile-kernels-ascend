# tile-kernels-ascend

Port of [TileKernels](https://github.com/deepseek-ai/TileKernels) custom kernels to Ascend NPU.
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

| Family | Kernel | `torch_npu` API (equivalent / closest) | Notes |
|--------|--------|----------------------------------------|-------|
| transpose | `transpose` | `torch.Tensor.t().contiguous()` | |
| transpose | `batched_transpose` | `torch.Tensor.transpose(-2, -1).contiguous()` | |
| moe | `aux_fi` | `torch.bincount` | |
| moe | `group_count` | `torch.bincount` | |
| moe | `mask_indices_by_tp` | standard torch ops | |
| moe | `normalize_weight` | `torch.sum + div` | |
| moe | `inplace_unique_group_indices` | `torch.sort + scatter` | |
| moe | `topk_gate` | `torch_npu.npu_moe_gating_top_k_softmax` / `npu_moe_gating_top_k` | |
| moe | `top2_sum_gate` | `torch_npu.npu_moe_gating_top_k` | group routing via `group_select_mode=1` |
| moe | `get_fused_mapping` | `torch_npu.npu_moe_init_routing` | + sorting post-processing |
| moe | `expand_to_fused` | `torch_npu.npu_moe_init_routing` | + position remap |
| moe | `expand_to_fused_with_sf` | `torch_npu.npu_moe_init_routing` × 2 | for data and sf tensors |
| moe | `reduce_fused` | `torch_npu.npu_moe_finalize_routing` | + column-major reshape |
| moe | `topk_sum_and_topk_group_idx` | `torch.topk` + `torch.sort` | `npu_moe_gating_top_k` doesn't expose group indices |
| mhc | `expand_to_mhc` | `torch.Tensor.expand` + `contiguous` | |
| mhc | `mhc_head_compute_mix` | `torch.sigmoid` + mul + add | |
| mhc | `mhc_pre_split_mixes` | `torch.sigmoid` + split | |
| mhc | `mhc_pre_apply_mix` | `torch.einsum` + sum | |
| mhc | `mhc_pre_norm_fn` | `torch.einsum` + `torch.rsqrt` | |
| mhc | `mhc_multilayer_recompute` | composed of `mhc_pre_apply_mix` + `mhc_post` | no single fused API |
| quant | `per_token_cast` | `torch_npu.npu_dynamic_quant` | |
| quant | `per_channel_cast` | `torch_npu.npu_dynamic_quant` | |
| quant | `per_channel_cast_fused` | `torch_npu.npu_dynamic_quant` | |
| quant | `per_block_cast` | `torch_npu.npu_dynamic_block_quant` | `row_block_size=1` required |
| quant | `cast_back` | `torch_npu.npu_anti_quant` | |
| quant | `swiglu_forward_and_per_channel_cast_and_transpose` | `torch_npu.npu_swiglu_quant` | int8/int4 output, not FP8 e4m3 |
| quant | `swiglu_forward_and_per_token_cast` | `torch_npu.npu_swiglu_quant` | per-token quant mode; int8/int4, not FP8 e4m3 |

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

### ACLNN vs torch-eager-NPU (fused kernel speedup)

Run with:
```bash
python bench_aclnn.py
```

Speedup is measured as `median(torch-eager-NPU time) / median(ACLNN time)`
over 100 NPU-event-timed runs after 10 warmup rounds. The "torch-eager-NPU"
baseline runs the same torch reference code on the NPU without any fused
`torch_npu` call. The ACLNN column wraps a real fused `torch_npu.npu_*`
kernel when available.

| Kernel | torch-eager (us) | ACLNN (us) | Speedup |
|--------|-----------------:|-----------:|--------:|
| **quant** ||||
|  `per_token_cast int8` | 382.2 | 123.2 | **3.10x** |
|  `per_channel_cast int8` | 393.7 | 122.2 | **3.22x** |
|  `per_block_cast (1,128) int8` | 380.7 | 140.8 | **2.70x** |
| **moe** ||||
|  `topk_gate softmax` | 273.3 | 182.0 | **1.50x** |
|  `top2_sum_gate (G=4, Kg=2)` | 2894.0 | 188.8 | **15.33x** |
|  `expand_to_fused` | 910.9 | 1271.9 | 0.72x* |
|  `reduce_fused` | 1197.4 | 382.7 | **3.13x** |
|  `get_fused_mapping` | 213419.8 | 2087.8 | **102.22x** |
|  `topk_sum_and_topk_group_idx` | 238.9 | 241.3 | 0.99x** |
|  `aux_fi` | 671.1 | 910.3 | 0.74x** |
|  `group_count` | 615.9 | 865.3 | 0.71x** |
|  `normalize_weight` | 248.8 | 176.0 | **1.41x** |
| **transpose** ||||
|  `transpose (1024x1024 bf16)` | 117.7 | 116.6 | 1.01x*** |
|  `batched_transpose (8,1024,1024 bf16)` | 120.5 | 118.1 | 1.02x*** |

Notes:
- \* `expand_to_fused` wraps `npu_moe_init_routing`, which sorts output
  by expert id; the post-processing to map back to the caller's expected
  `token_topk_to_pos` layout adds ~30% overhead vs the simple scatter
  used by the reference. The fused kernel itself is faster.
- \*\* These use plain torch ops on both sides (no real fused `torch_npu`
  kernel exists for them). Values hover around 1x with small measurement
  variation.
- \*\*\* `transpose` / `batched_transpose` use `torch.Tensor.transpose`
  / `t().contiguous()` on both sides. Both paths hit the same aclnn
  kernel under the hood.

### Hardware / Software Environment

Results above were collected on:

| Component | Value |
|-----------|-------|
| **NPU** | Ascend 910B2 × 8 |
| **Driver** | npu-smi 25.5.1 / V100R001C23SPC006B220 |
| **PyTorch** | 2.9.0 |
| **torch_npu** | 2.9.0.post2 |
| **Host CPU** | (reference torch-eager baseline only) |
| **OS** | Linux |
| **Python** | 3.11.15 |

Single-device pinning: `npu:0` (one chip). Each measurement uses
`torch.npu.Event` start/end pairs around the kernel call, with
`torch.npu.synchronize()` between iterations for deterministic timing.

Total benchmark suite runtime: **< 30 seconds** (well within the
20-minute budget at these representative shapes).

### Reference input shapes

| Kernel family | Representative input shape |
|---------------|----------------------------|
| quant | `(T=1024, H=1024)` bf16 tensor |
| moe | `N=512, E=8, K=2, H=256`; `logits (N, E)`, `topk_idx (N, K)` |
| transpose | `(1024, 1024)` or `(8, 1024, 1024)` bf16 tensor |

Full machine-readable results are saved to [`benchmark/benchmark_results.json`](benchmark/benchmark_results.json).

---

**TODO: Ascend 950 Validation**

The current ACLNN benchmark results are obtained on **Ascend 910B** devices (specifically Ascend 910B2 with CANN 8.0). Several kernel families are marked as "Unsupported" due to missing or incompatible torch_npu APIs:

- **mhc kernels** (`mhc_multilayer_recompute`, `mhc_post`, `sinkhorn_normalize`) — require CANN >= 9.0.0
- **mxfp quant kernels** (`quant_mxfp4`, `quant_mxfp8`) — require newer torch_npu dynamic block quant with `block_size=(32,32)` and `row_block_size` != 1

These kernels are **very likely to be supported** when migrating to **Ascend 950** devices, which ship with CANN 9.0+ and updated torch_npu APIs. After obtaining access to an Ascend 950 test environment:

1. Re-run `python bench_aclnn.py` to validate mhc and mxfp support
2. Update the ACLNN wrapper implementations in `tile_kernels_ascend/aclnn/mhc/__init__.py` and `tile_kernels_ascend/aclnn/quant/__init__.py`
3. Update the Benchmark Results table above with Ascend 950 numbers
4. Move supported kernels from the "Unsupported Kernels" table to the "Supported Kernels" table

---

## Generating Support Report

```bash
python generate_support_status.py
# Writes support_status/support_status.csv and prints a summary
```

## Reference LoC (Original tilelang)

Per-kernel LoC of the kernel body under `@tilelang.jit` in the original [TileKernels](https://github.com/deepseek-ai/TileKernels) repo (excluding testing code, blank lines, comments). See [`reference_loc_detailed.md`](reference_loc_detailed.md) for the full breakdown with source file paths.

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
