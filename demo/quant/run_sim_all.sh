#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEMOS=(
  demo_npu_dynamic_quant.py
  demo_npu_anti_quant.py
  demo_npu_dynamic_block_quant.py
  demo_npu_swiglu_quant.py
)

for demo in "${DEMOS[@]}"; do
  bash "${SCRIPT_DIR}/run_sim.sh" "${demo}"
  echo ""
done

echo "All simulator runs completed."
