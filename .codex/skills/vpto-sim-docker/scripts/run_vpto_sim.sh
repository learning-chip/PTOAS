#!/usr/bin/env bash
# Minimal VPTO simulator runner for agent_npu_cann_950 Docker image.
# CPU-host only (DEVICE=SIM); no NPU device required.

set -euo pipefail

PTO_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-/llvm-workspace/llvm-project/build-shared}"
PTO_INSTALL_DIR="${PTO_INSTALL_DIR:-${PTO_SOURCE_DIR}/install}"
WORK_ROOT="${WORK_ROOT:-${PTO_SOURCE_DIR}/.work}"
WORK_SPACE="${WORK_SPACE:-${WORK_ROOT}/vpto-sim}"
CASE_NAME="${CASE_NAME:-}"
CASE_PREFIX="${CASE_PREFIX:-}"
JOBS="${JOBS:-4}"
DEVICE="${DEVICE:-SIM}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--case NAME | --prefix PREFIX] [--jobs N] [--workspace DIR]

Examples:
  $(basename "$0") --case micro-op/binary-vector/vadd
  $(basename "$0") --prefix micro-op/binary-vector --jobs 4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) CASE_NAME="$2"; shift 2 ;;
    --prefix) CASE_PREFIX="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --workspace) WORK_SPACE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -n "$CASE_NAME" && -n "$CASE_PREFIX" ]]; then
  echo "error: use --case or --prefix, not both" >&2
  exit 1
fi

PTOAS_BIN="${PTOAS_BIN:-${PTO_SOURCE_DIR}/build/tools/ptoas/ptoas}"
[[ -x "$PTOAS_BIN" ]] || {
  echo "error: ptoas not built; run quick_install.sh first" >&2
  exit 1
}

export PTO_SOURCE_DIR LLVM_BUILD_DIR PTO_INSTALL_DIR
export PATH="${PTO_SOURCE_DIR}/build/tools/ptoas:${PATH}"
export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-$(
  find /usr/local/Ascend -maxdepth 2 -type d -name 'cann*' 2>/dev/null | sort | head -1
)}"
[[ -n "$ASCEND_HOME_PATH" ]] || { echo "error: ASCEND_HOME_PATH not found" >&2; exit 1; }
[[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]] && source "${ASCEND_HOME_PATH}/set_env.sh"

if [[ -z "${SIM_LIB_DIR:-}" ]]; then
  CAMODEL_SRC="$(find "${ASCEND_HOME_PATH}" -type d -path '*/simulator/dav_3510/lib' | sort | head -1)"
  [[ -n "$CAMODEL_SRC" ]] || { echo "error: dav_3510 simulator lib not found under ${ASCEND_HOME_PATH}" >&2; exit 1; }
  SIM_LIB_DIR="$(python3 "${PTO_SOURCE_DIR}/scripts/prepare_quiet_camodel.py" \
    --source-dir "${CAMODEL_SRC}" \
    --output-dir "${WORK_ROOT}/camodel")"
fi

mkdir -p "${WORK_SPACE}"

export WORK_SPACE ASCEND_HOME_PATH PTOAS_BIN SIM_LIB_DIR DEVICE

if [[ -n "$CASE_NAME" ]]; then
  export CASE_NAME
  bash "${PTO_SOURCE_DIR}/test/vpto/scripts/run_host_vpto_validation.sh"
else
  export CASE_PREFIX JOBS
  bash "${PTO_SOURCE_DIR}/test/vpto/scripts/run_host_vpto_validation_parallel.sh"
fi
