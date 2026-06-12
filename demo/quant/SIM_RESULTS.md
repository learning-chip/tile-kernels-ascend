# CANN Simulator Results for Quant Demos

This file documents simulator runs of the torch_npu quant API demos
on **Ascend950** using `cannsim` (CPU-only host, no NPU hardware).

## Quick Start

Run a single demo file under the simulator:

```bash
source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash
bash run_sim.sh demo_npu_dynamic_quant.py
```

Run all four demo files in sequence:

```bash
bash run_sim_all.sh
```

Results are written to `demo/quant/sim_outputs/<demo_basename>/`.
The directory is git-ignored; see the `.gitignore` at the repo root.

## Reproduce Commands

Each run invokes:

```bash
cannsim record ./run_sim_entry.sh \
  -s Ascend950 --gen-report \
  -o <output_dir> \
  -u <demo_script.py>
```

`run_sim_entry.sh` is a thin wrapper (`exec python3 "$@"`) that cannsim
launches as the user application. The demo Python path is forwarded via
`-u` and becomes `$1` inside the wrapper.

## Results Summary

### Overall

| Demo file | Simulator wall-time | Total kernel span | Max SoC cycles |
|-----------|------------------:|-----------------:|--------------:|
| `demo_npu_dynamic_quant.py`      | 152 s | 124,350 cycles | 420 (1.26 s) |
| `demo_npu_anti_quant.py`         | 134 s | 132,092 cycles | 419 (1.27 s) |
| `demo_npu_dynamic_block_quant.py`| 187 s | 118,513 cycles | 420 (1.27 s) |
| `demo_npu_swiglu_quant.py`       | 201 s | 181,217 cycles | 420 (1.26 s) |
| **Total**                        | 674 s (11.2 min) |             |               |

All 24 active demos executed successfully (printed expected output).
4 demos SKIPPED (int4/3-D/HiFLOAT8/FP8-e5m2 unsupported on Ascend950).

### Per-Demo Kernel Times

Span = total simulated AI-Core cycles from first launch start to last launch end.
MaxDur = longest single-kernel execution on any vector core.
#Launches = total AI-Core kernel launches dispatched by the demo.
Cores = max vector cores used concurrently by any single launch.

#### demo_npu_dynamic_quant.py

| Demo                                | Span      | MaxDur  | #Launches | Cores |
|-------------------------------------|----------:|--------:|----------:|------:|
| per-token INT8 fp16    (4,32,fp16)  |   7,041   |  6,960  |        4  |     4 |
| per-token INT8 bf16    (4,32,bf16)  |  10,442   |  4,216  |        3  |     1 |
| smooth-quant           (8,64,fp16)  |   8,154   |  4,319  |        6  |     4 |
| moe-grouped smooth     (12,32,3grp) |  32,921   |  5,677  |       32  |    12 |
| per-channel INT8       (128,128,bf16)|   9,916  |  6,620  |       96  |    64 |
| per-channel+smooth     (64,64,bf16) |  38,997   |  7,759  |      214  |    64 |
| int4 quint4x2                       |   SKIPPED |         |           |       |
| 3-D bf16 input                      |   SKIPPED |         |           |       |

#### demo_npu_anti_quant.py

| Demo                                | Span      | MaxDur  | #Launches | Cores |
|-------------------------------------|----------:|--------:|----------:|------:|
| int8->fp16             (8,32)       |  63,672   |  4,115  |       29  |     1 |
| int8->bf16             (8,32)       |  12,232   |  3,497  |        6  |     1 |
| asymmetric dequant     (4,16)       |   4,374   |  4,374  |        1  |     1 |
| roundtrip per-channel  (4,32)       |  11,928   |  5,952  |       66  |    64 |
| src_dtype hint         (4,16)       |  31,617   |  4,777  |       16  |     1 |

#### demo_npu_dynamic_block_quant.py

| Demo                                | Span      | MaxDur  | #Launches | Cores |
|-------------------------------------|----------:|--------:|----------:|------:|
| FP8 per-token          (128,128,bf16)|  11,123  | 11,040  |       64  |    64 |
| FP8 per-block          (256,256,128x128)|  7,499 | 4,011  |       64  |    32 |
| FP8 large hidden       (16,512,col=128)| 30,477 | 10,451  |      196  |    64 |
| FP8 row_bs=1,col=128   (64,128,fp16)|  21,508   |  8,421  |      160  |    64 |
| FP8 min_scale          (4,128,bf16) |  47,180   |  9,504  |      114  |    64 |
| HiFLOAT8 (dst_type=129)             |   SKIPPED |         |           |       |
| FP8 e5m2 (dst_type=130)             |   SKIPPED |         |           |       |

#### demo_npu_swiglu_quant.py

| Demo                                | Span      | MaxDur  | #Launches | Cores |
|-------------------------------------|----------:|--------:|----------:|------:|
| static/per-channel     (8,64,bf16)  |  15,572   |  9,281  |       17  |     8 |
| dynamic/per-token      (16,128,bf16)|  20,922   |  9,841  |       40  |    16 |
| MoE grouped            (32,128,4grp)|  31,793   | 13,331  |       69  |    32 |
| static+groups          (24,64,3grp) |  31,864   | 10,203  |       50  |    24 |
| count group_list       (16,64,4grp) |  23,671   |  6,640  |       32  |    16 |
| right activation       (8,64,bf16)  |  12,708   |  5,547  |       12  |     8 |
| int4 output            (8,128,fp32) |  13,372   |  5,426  |       20  |     8 |
| fp16 input             (8,64,fp16)  |  17,299   |  6,802  |       12  |     8 |

## Notes

- Simulator wall-time is dominated by ~35 s startup overhead per cannsim invocation;
  actual kernel simulation is fast for these small tensor sizes.
- `cannsim` exits with code 1 due to a Python-shutdown segfault AFTER all kernels
  complete. The kernels themselves execute successfully (output is printed).
- "Span", "MaxDur", and cycle counts are extracted from the instruction-level
  logs in `<output_dir>/cannsim_*/log_ca/core*.veccore*.instr_log.dump`.
- To convert cycles to predicted NPU wall-time, multiply by the Ascend950
  clock period (chip frequency is version-specific).
- Demo-to-launch-group mapping is heuristic, based on temporal gaps (>3000 cycles)
  between consecutive AI-Core kernel launches and `blkDim` patterns.
  For exact attribution, inspect the raw `instr_log.dump` files.
- Use `--gen-report` with `cannsim record` to also generate
  `trace_core*.json` files for Chrome-trace visualization (requires the
  simulator to finish cleanly; currently blocked by the exit segfault).
