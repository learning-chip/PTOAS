#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${HERE}/build"
mkdir -p "${OUT}"
: "${ASCEND_HOME_PATH:?source CANN set_env.sh first}"
BISHENG="${BISHENG:-${ASCEND_HOME_PATH}/bin/bisheng}"
"${BISHENG}" -O3 -std=gnu++17 -fPIC -Wno-macro-redefined \
  -Wno-ignored-attributes -Wno-unknown-attributes \
  --cce-aicore-arch=dav-c310-vec -xcce -Xhost-start -Xhost-end \
  -I"${ASCEND_HOME_PATH}/include" -I"${HERE}" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/asc/include" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/asc/include/basic_api" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/asc" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/asc/impl" \
  -c "${HERE}/reference_compact_store.cpp" -o "${OUT}/reference_compact_store.o"
"${BISHENG}" --cce-fatobj-link -shared -fPIC -Wl,--no-undefined \
  "${OUT}/reference_compact_store.o" -L"${ASCEND_HOME_PATH}/lib64" \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" -Wl,--no-as-needed -lruntime \
  -o "${OUT}/libreference_compact_store.so"
printf '%s\n' "${OUT}/libreference_compact_store.so"
