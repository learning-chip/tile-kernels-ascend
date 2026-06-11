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
# Default tests (fast subset) — finishes in ~5 minutes
pytest tests/ -q

# Full benchmark run (requires --run-benchmark)
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

- PyTorch + torch_npu (2.9.0+)
- NPU: Ascend 910B2 × 8
- CPU golden reference: PyTorch eager on CPU
- NPU mock kernel: same PyTorch eager code on NPU
