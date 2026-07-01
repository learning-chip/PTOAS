---
name: vpto-sim-docker
description: Run PTOAS VPTO simulator validation (DEVICE=SIM, CPU-host only) inside the agent_npu_cann_950 Docker image with prebuilt vpto-dev LLVM. Use when asked to run VPTO sim tests, camodel validation, test/vpto cases, or micro-op/binary-vector inside Docker without NPU devices.
---

# VPTO Simulator Mode in Docker

CPU-host VPTO validation using CANN's `dav_3510` camodel. No NPU device mounts required.

## Prerequisites

1. **Docker image** `agent_npu_cann_950:9.0.0` from
   [learning-chip/agent_docker_npu PR #11](https://github.com/learning-chip/agent_docker_npu/pull/11)
   (`.devcontainer/kernel_dev_950/Dockerfile`). Image pre-builds
   `vpto-dev/llvm-project` branch `feature-vpto` at
   `/llvm-workspace/llvm-project/build-shared`.

2. **ptoas built inside the container** (once per checkout):

```bash
export LLVM_BUILD_DIR=/llvm-workspace/llvm-project/build-shared
bash quick_install.sh
```

## Host: enter container (sim-only)

```bash
HOST_MOUNT_DIR=$HOME/work_code/workdir_for_agent   # adjust to your layout

docker run --rm -it \
  -v "${HOST_MOUNT_DIR}:/workdir" \
  -w /workdir/ptoas_fork \
  agent_npu_cann_950:9.0.0 bash
```

## Inside container: one command

From the PTOAS repo root:

```bash
# single case
bash .codex/skills/vpto-sim-docker/scripts/run_vpto_sim.sh \
  --case micro-op/binary-vector/vadd

# full binary-vector suite (9 cases)
bash .codex/skills/vpto-sim-docker/scripts/run_vpto_sim.sh \
  --prefix micro-op/binary-vector --jobs 4
```

## Inside container: manual steps

Match `test/vpto/scripts/run_host_vpto_validation*.sh` and `.github/workflows/ci_sim.yml`.

```bash
export PTO_SOURCE_DIR=$PWD
export LLVM_BUILD_DIR=/llvm-workspace/llvm-project/build-shared
export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install
export PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH
export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}

ASCEND_HOME_PATH="$(find /usr/local/Ascend -maxdepth 2 -type d -name 'cann*' | sort | head -1)"
source "$ASCEND_HOME_PATH/set_env.sh"

CAMODEL_SRC="$(find "$ASCEND_HOME_PATH" -type d -path '*/simulator/dav_3510/lib' | sort | head -1)"
SIM_LIB_DIR="$(python3 scripts/prepare_quiet_camodel.py \
  --source-dir "$CAMODEL_SRC" \
  --output-dir "$PTO_SOURCE_DIR/.work/camodel")"

WORK_SPACE=$PTO_SOURCE_DIR/.work/vpto-sim \
ASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
PTOAS_BIN=$PTO_SOURCE_DIR/build/tools/ptoas/ptoas \
SIM_LIB_DIR="$SIM_LIB_DIR" \
DEVICE=SIM \
CASE_PREFIX=micro-op/binary-vector \
JOBS=4 \
bash test/vpto/scripts/run_host_vpto_validation_parallel.sh
```

## Required env (defaults in helper script)

| Variable | Value |
|---|---|
| `DEVICE` | `SIM` |
| `PTOAS_BIN` | `$PWD/build/tools/ptoas/ptoas` |
| `PTOAS_FLAGS` | `--pto-arch a5 --pto-backend=vpto` (script default) |
| `SIM_LIB_DIR` | quiet camodel dir from `scripts/prepare_quiet_camodel.py` |
| `WORK_SPACE` | `$PWD/.work/vpto-sim` (scratch output) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `unknown target CPU 'znver4'` | Rebuild ptoas from current repo (uses `x86-64` host CPU), or `export PTOAS_HOST_TARGET_CPU=x86-64` |
| `bisheng not found` | `source "$ASCEND_HOME_PATH/set_env.sh"` |
| `no dav_3510 simulator lib` | CANN 950 image missing sim libs; verify `agent_npu_cann_950:9.0.0` base |
| `PTOAS_BIN is not executable` | Run `quick_install.sh` first |

## Success criteria

- Serial: `All N VPTO case(s) passed`
- Parallel: `PASS=N FAIL=0` in `.work/vpto-sim/parallel-summary.tsv`
