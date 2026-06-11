# ACLNN Backend: torch.* Operations Already Use NPU-Native Kernels

## Executive Summary

All `torch.*` operations in the ACLNN backend already execute on NPU-native kernels through PyTorch's automatic dispatch mechanism. **No manual replacements with `torch_npu.*` APIs are needed or beneficial.**

## Audit Results

### Operations Audited

| Operation | Location | Time (μs) | NPU-Native Alternative | Conclusion |
|-----------|----------|-----------|------------------------|------------|
| `torch.sort` (stable, int32) | moe backend (3 calls) | 347 | `npu_sort_v2` | ✅ torch.sort is optimal |
| `torch.sigmoid` | mhc backend | 89 | None exists | ✅ torch.sigmoid is optimal |
| `torch.cat` | mhc backend | 111 | None exists | ✅ torch.cat is optimal |
| `torch.einsum` | mhc backend | 201 | None exists | ✅ torch.einsum is optimal |
| `torch.bincount` | moe backend | 574 | None exists | ✅ torch.bincount is optimal |

**Benchmark environment:** Ascend 910B2, torch_npu 2.9.0, 100 iterations with 20 warmup runs

## Why torch.* is Already Optimal

### PyTorch Dispatch Mechanism

When you create a tensor on NPU device:
```python
x = torch.randn(512, 8, device='npu')
```

All subsequent `torch.*` operations on that tensor automatically dispatch to NPU-native kernels:
```python
vals, idx = torch.sort(x, dim=1, stable=True)  # → aclnn sort kernel
```

This is not a "fallback" or "emulation" — it's a direct call to optimized NPU kernels.

### Case Study: torch.sort vs npu_sort_v2

**torch.sort** (current implementation):
- Signature: `torch.sort(input, dim=-1, descending=False, stable=False)`
- Returns: `(sorted_values, sorted_indices)`
- Supports: int32, float32, float16, bfloat16
- Performance: 347 μs per call
- Status: ✅ Active, recommended

**npu_sort_v2** (would-be alternative):
- Signature: `torch_npu.npu_sort_v2(self, dim=-1, descending=False)`
- Returns: `sorted_values` only (no indices)
- Supports: float32, float16, bfloat16 (NOT int32)
- Performance: Similar to torch.sort
- Status: ⚠️ **Deprecated** in torch_npu documentation

**Why torch.sort wins:**
1. Returns indices (required for `get_fused_mapping`, `inplace_unique_group_indices`)
2. Supports `stable=True` (required for deterministic routing)
3. Supports int32 (required by our kernels)
4. Not deprecated

### Why Other Operations Have No Alternatives

| Operation | Reason for No torch_npu.* Alternative |
|-----------|---------------------------------------|
| `torch.sigmoid` | Fundamental element-wise op, torch_npu doesn't provide separate API |
| `torch.cat` | Fundamental concatenation op, torch_npu doesn't provide separate API |
| `torch.einsum` | High-level tensor contraction, decomposes to matmul internally |
| `torch.bincount` | Specialized histogram op, torch_npu doesn't provide separate API |

## Benchmark Methodology

```python
def timer_us(fn, warmup=10, repeat=100):
    """Measure median execution time over `repeat` runs."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    
    times = []
    for _ in range(repeat):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    
    times.sort()
    return times[len(times) // 2]
```

## Recommendations

**Do NOT replace torch.* with torch_npu.* operations** in the ACLNN backend. The current implementation is already optimal.

**Exception:** For operations with explicit fused kernel support (like `npu_moe_gating_top_k_softmax`, `npu_dynamic_quant`), continue using `torch_npu.*` directly — these provide significant speedups over decomposed operations.

## Related Documentation

- [Benchmark Results](benchmark/benchmark_results.json) — Performance of fused torch_npu vs torch-eager baseline
- [ACLNN Backend Implementation](tile_kernels_ascend/aclnn/) — Source code
- [README.md](README.md#benchmarks) — High-level benchmarks section

## Testing

To reproduce this audit:
```bash
python bench_native_ops.py
```

Expected output: All torch.* operations show competitive performance with no faster torch_npu.* alternatives available.
