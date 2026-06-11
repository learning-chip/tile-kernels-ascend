# Reference LoC — Original TileLang @jit Kernels

Line counts of the kernel body under `@tilelang.jit` in the original TileKernels repo (not including testing code, blank lines, or comments).

## engram (5 kernels, 493 LoC)

| Kernel function | Source file | LoC |
|-----------------|-------------|----:|
| `get_engram_grad_w_reduce_kernel` | `engram/engram_grad_w_reduce_kernel.py` | 49 |
| `get_engram_fused_weight_kernel` | `engram/engram_fused_weight_kernel.py` | 27 |
| `get_engram_hash_kernel` | `engram/engram_hash_kernel.py` | 50 |
| `get_engram_gate_fwd_kernel` | `engram/engram_gate_kernel.py` | 140 |
| `get_engram_gate_bwd_kernel` | `engram/engram_gate_kernel.py` | 227 |

## mhc (20 kernels, 961 LoC)

| Kernel function | Source file | LoC |
|-----------------|-------------|----:|
| `expand_to_mhc_fwd_tl` | `mhc/expand_kernel.py` | 23 |
| `expand_to_mhc_bwd_tl` | `mhc/expand_kernel.py` | 24 |
| `_mhc_multilayer_recompute_kernel` | `mhc/multilayer_recompute_kernel.py` | 91 |
| `_mhc_pre_split_mixes_fwd` | `mhc/pre_split_mixes_kernel.py` | 49 |
| `_mhc_pre_split_mixes_bwd` | `mhc/pre_split_mixes_kernel.py` | 75 |
| `_mhc_post_fwd` | `mhc/post_kernel.py` | 43 |
| `_mhc_post_bwd` | `mhc/post_kernel.py` | 71 |
| `_mhc_sinkhorn_fwd` | `mhc/sinkhorn_kernel.py` | 37 |
| `_mhc_sinkhorn_bwd` | `mhc/sinkhorn_kernel.py` | 81 |
| `_mhc_pre_apply_mix_fwd` | `mhc/pre_apply_mix_kernel.py` | 34 |
| `_mhc_pre_apply_mix_bwd` | `mhc/pre_apply_mix_kernel.py` | 44 |
| `_mhc_head_compute_mix_fwd` | `mhc/head_compute_mix_kernel.py` | 20 |
| `_mhc_head_compute_mix_bwd` | `mhc/head_compute_mix_kernel.py` | 45 |
| `_mhc_pre_big_fuse` | `mhc/pre_big_fuse_kernel.py` | 96 |
| `_mhc_fn_normw_merge_fwd` | `mhc/norm_fn_kernel.py` | 16 |
| `_mhc_fn_normw_merge_bwd` | `mhc/norm_fn_kernel.py` | 26 |
| `_mhc_pre_norm_fn_fwd_mul` | `mhc/norm_fn_kernel.py` | 53 |
| `_mhc_pre_norm_fn_fwd_norm` | `mhc/norm_fn_kernel.py` | 38 |
| `_mhc_pre_norm_fn_bwd_norm` | `mhc/norm_fn_kernel.py` | 31 |
| `_mhc_pre_norm_fn_bwd_mul` | `mhc/norm_fn_kernel.py` | 64 |

## moe (11 kernels, 713 LoC)

| Kernel function | Source file | LoC |
|-----------------|-------------|----:|
| `get_normalize_weight_kernel` | `moe/normalize_weight_kernel.py` | 29 |
| `get_group_count_kernel` | `moe/group_count_kernel.py` | 32 |
| `get_reduce_fused_kernel` | `moe/reduce_fused_kernel.py` | 52 |
| `get_expand_to_fused_kernel` | `moe/expand_to_fused_kernel.py` | 70 |
| `get_topk_gate_kernel` | `moe/topk_gate_kernel.py` | 40 |
| `get_mask_indices_by_tp_kernel` | `moe/mask_indices_by_tp_kernel.py` | 34 |
| `get_inplace_unique_group_indices_kernel` | `moe/inplace_unique_group_indices_kernel.py` | 34 |
| `get_top2_sum_gate_kernel` | `moe/top2_sum_gate_kernel.py` | 221 |
| `get_aux_fi_kernel` | `moe/aux_fi_kernel.py` | 33 |
| `get_get_fused_mapping_kernel` | `moe/get_fused_mapping_kernel.py` | 120 |
| `get_topk_sum_and_topk_group_idx_kernel` | `moe/topk_sum_and_topk_group_idx_kernel.py` | 48 |

## quant (11 kernels, 1,048 LoC)

| Kernel function | Source file | LoC |
|-----------------|-------------|----:|
| `get_per_channel_cast_fused_kernel` | `quant/per_channel_cast_fused_kernel.py` | 98 |
| `get_per_token_cast_kernel` | `quant/per_token_cast_kernel.py` | 113 |
| `get_swiglu_forward_and_per_channel_cast_and_transpose_kernel` | `quant/swiglu_forward_and_per_channel_cast_and_transpose_kernel.py` | 139 |
| `get_cast_back_kernel` | `quant/cast_back_kernel.py` | 46 |
| `get_per_channel_cast_and_transpose_kernel` | `quant/per_channel_cast_and_transpose_kernel.py` | 57 |
| `get_swiglu_backward_and_per_token_cast_kernel` | `quant/swiglu_backward_and_per_token_cast_kernel.py` | 120 |
| `get_per_block_cast_kernel` | `quant/per_block_cast_kernel.py` | 112 |
| `get_per_token_cast_to_e5m6_kernel` | `quant/per_token_cast_to_e5m6_kernel.py` | 69 |
| `get_per_block_cast_lossless_kernel` | `quant/per_block_cast_lossless_kernel.py` | 96 |
| `get_swiglu_forward_and_per_token_cast_kernel` | `quant/swiglu_forward_and_per_token_cast_kernel.py` | 134 |
| `get_cast_back_e5m6_kernel` | `quant/cast_back_e5m6_kernel.py` | 64 |

## transpose (1 kernel, 46 LoC)

| Kernel function | Source file | LoC |
|-----------------|-------------|----:|
| `get_batched_transpose_kernel` | `transpose/batched_transpose_kernel.py` | 46 |

---

## Grand Total

| Family | Kernels | LoC |
|--------|--------:|----:|
| engram | 5 | 493 |
| mhc | 20 | 961 |
| moe | 11 | 713 |
| quant | 11 | 1,048 |
| transpose | 1 | 46 |
| **Total** | **48** | **3,261** |
