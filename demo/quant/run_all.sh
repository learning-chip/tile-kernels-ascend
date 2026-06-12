#!/usr/bin/env bash
# Run all quant demos for Ascend NPU.
# Usage:  bash demo/quant/run_all.sh
#
# Each demo is independent and can also be run individually, e.g.:
#   source /usr/local/Ascend/cann-9.0.0/set_env.sh
#   task-submit --device 1 --run "python demo/quant/demo_npu_dynamic_quant.py"

set -e

# Source CANN env if libhccl.so is not already on LD_LIBRARY_PATH
if ! echo "$LD_LIBRARY_PATH" | grep -q "cann"; then
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEMOS=(
  demo_npu_dynamic_quant.py
  demo_npu_anti_quant.py
  demo_npu_dynamic_block_quant.py
  demo_npu_swiglu_quant.py
)

for demo in "${DEMOS[@]}"; do
  echo "============================================"
  echo "Running: $demo"
  echo "============================================"
  task-submit --device 1 --run "python $SCRIPT_DIR/$demo"
  echo ""
done

echo "All demos completed."
