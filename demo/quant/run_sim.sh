#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"
source "${ASCEND_HOME_PATH}/bin/setenv.bash"

usage() {
  cat <<'EOF'
Usage: run_sim.sh <demo_file> [output_dir]
  demo_file   : Python demo script in this directory (e.g. demo_npu_dynamic_quant.py)
  output_dir  : Simulator output directory (default: ./sim_outputs/<demo_basename>)

Examples:
  bash run_sim.sh demo_npu_dynamic_quant.py
  bash run_sim.sh demo_npu_anti_quant.py ./my_output

  # Run all four demo files sequentially
  for f in demo_npu_dynamic_quant.py demo_npu_anti_quant.py \
           demo_npu_dynamic_block_quant.py demo_npu_swiglu_quant.py; do
    bash run_sim.sh "$f"
  done
EOF
  exit 1
}

DEMO_FILE="${1:-}"
[ -z "${DEMO_FILE}" ] && usage
DEMO_PATH="${SCRIPT_DIR}/${DEMO_FILE}"
[ ! -f "${DEMO_PATH}" ] && { echo "ERROR: ${DEMO_FILE} not found in ${SCRIPT_DIR}"; usage; }

BASENAME="${DEMO_FILE%.py}"
OUT_DIR="${2:-${SCRIPT_DIR}/sim_outputs/${BASENAME}}"
mkdir -p "${OUT_DIR}"

echo "============================================"
echo "cannsim record: ${DEMO_FILE}"
echo "output: ${OUT_DIR}"
echo "============================================"

set +e
cannsim record \
  "${SCRIPT_DIR}/run_sim_entry.sh" \
  -s Ascend950 \
  --gen-report \
  -o "${OUT_DIR}" \
  -u "${DEMO_PATH}"
rc=$?
set -e

echo ""
echo "cannsim exit code: ${rc}"
echo "Output dir: ${OUT_DIR}"

if [ "${rc}" -ne 0 ]; then
  echo "NOTE: cannsim returned ${rc}. This is typically caused by a segfault"
  echo "      during Python shutdown AFTER the kernels finished successfully."
  echo "      Check the cannsim.log for output lines to confirm kernel execution."
fi

echo ""
echo "--- Simulator-predicted kernel times (from cannsim.log) ---"
CANNSIM_LOG="$(find "${OUT_DIR}" -name cannsim.log -maxdepth 2 | head -1)"
if [ -n "${CANNSIM_LOG}" ] && [ -f "${CANNSIM_LOG}" ]; then
  grep -E "(parallel simulation finish|all tasks are finished|\[.*\].*out\.shape|\[.*\].*SKIPPED)" "${CANNSIM_LOG}" || true
else
  echo "(no cannsim.log found)"
fi
