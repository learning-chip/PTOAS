#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
DEVICE="${REPRO_DEVICE:-${2:-0}}"
export REPRO_DEVICE="${DEVICE}"
export PYTHONPATH="${REPRO_PYTHONPATH:-${REPO}/python:${REPO}/ptodsl}"
export PTOAS_BIN="${PTOAS_BIN:-$(command -v ptoas || true)}"
case "${1:-help}" in
  vmi-compile)
    exec python3 "${HERE}/desired_compact_store.py" --compile-only
    ;;
  vmi)
    set +e
    python3 "${HERE}/desired_compact_store.py" --device "npu:${DEVICE}"
    rc=$?
    set -e
    if [[ "${rc}" -ne 2 ]]; then
      echo "expected VMI runtime failure (exit 2), got ${rc}" >&2
      exit 1
    fi
    ;;
  asc)
    lib="$("${HERE}/build_reference.sh")"
    python3 "${HERE}/run_reference.py" --library "${lib}" --device "npu:${DEVICE}"
    ;;
  *)
    echo "usage: $0 {vmi-compile|vmi|asc} [device]" >&2
    exit 64
    ;;
esac
