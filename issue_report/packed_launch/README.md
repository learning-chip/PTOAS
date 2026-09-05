# TileLang-free packed UE8M0 launch bug

This reproducer contains only low-level PTODSL and ASC code. It does not import
TileLang, TileKernels, or any framework quantization helper. The packed UE8M0
failure is reduced to the compact grouped-store contract used when writing
packed scale bytes. Smaller byte-only forms either passed or were rejected by
PTOAS's frontend legality table and are not the failing pattern.

## Bug pattern

`desired_compact_store.py` creates one host-launchable A5 VPTO kernel. The kernel:

1. DMA-copies 24 FP32 values from GM to UB;
2. loads each value with `dist_mode="brc"` into a 64-lane VMI broadcast value;
3. performs multiply/add arithmetic;
4. stores each result with `group=1, stride=1` (one compact value per group);
5. DMA-copies the compact output back to GM.

There is no reduction, FP4 conversion, persistent loop, dynamic shape, or
per-block quantization logic. The PTODSL frontend emits legal VMI MLIR and the
script prints `PTODSL_COMPILE_OK` before launching. PTOAS lowers the compact
`group=1` store to a dense VPTO store without ASC `ONEPT_B32` placement; the
observed failure is after launch at `torch.npu.synchronize()` with ACL error
`507035`.

Compile-only (frontend syntax/type/lowering check):

```bash
PYTHONPATH="python:ptodsl" python3 issue_report/packed_launch/desired_compact_store.py --emit-mlir
```

On a real NPU, use `task-submit` and CANN 9.1.0:

```bash
task-submit --device 1 --run '\
set +u
source /usr/local/Ascend/canns/9.1.0/cann-9.1.0/set_env.sh
source /home/jzhuang/miniconda/etc/profile.d/conda.sh
conda activate cann91_dev
set -u
cd /home/jzhuang/work_dir/vmi_work_0827/ptoas_packed_launch_bug
export PTOAS_BIN="${PTOAS_BIN:-$(command -v ptoas)}"
export REPRO_DEVICE=1
export PYTHONPATH="$PWD/python:$PWD/ptodsl"
python3 issue_report/packed_launch/desired_compact_store.py --device 1
'
```

Expected VMI output:

```text
PTODSL_COMPILE_OK: VMI frontend accepted and compiled
VMI_RUNTIME_ERROR: ... device error type 3, error code is 507035
```

The frontend therefore accepts the VMI syntax and types; the failure is in the
lower-level VPTO/device execution of this packed grouped-store pattern.

## ASC reference

`reference_compact_store.cpp` uses the identical compact arithmetic and launch
shape. Its ASC implementation uses `BRC_B32` loads and `ONEPT_B32` stores. It performs
the equivalent compact store using ASC GM↔UB DMA primitives, launches with
`<<<1, nullptr, stream>>>`, synchronizes with ACL, and checks all 64 bytes.

Compile the ASC device/host source after sourcing CANN:

```bash
source /usr/local/Ascend/canns/9.1.0/cann-9.1.0/set_env.sh
bisheng -std=c++17 -O2 -fPIC -xcce --cce-aicore-arch=dav-c310-vec \
  -I"${ASCEND_HOME_PATH}/include" -c \
  issue_report/packed_launch/reference_compact_store.cpp \
  -o /tmp/reference_compact_store.o
```

For a complete executable, link the object with the CANN ACL/runtime flags
used by the host on the validation machine (for example `-L${ASCEND_HOME_PATH}/lib64
-lruntime -lascendcl -Wl,--cce-fatobj-link`). The expected output is:

```text
ASC_LAUNCH_OK: packed bytes matched
```

This reference compiles, launches, synchronizes, and preserves the packed byte
sequence, demonstrating that the ASC baseline and ABI are valid.

## Reproducer commands

```bash
./issue_report/packed_launch/run_repro.sh vmi-compile
./issue_report/packed_launch/run_repro.sh vmi 1       # expected ACL 507035
./issue_report/packed_launch/run_repro.sh asc 1       # compile/link/run ASC source
```

The VMI command must report compile success followed by the runtime ACL error;
any Python/PTODSL syntax or type error is a different failure and does not
reproduce this issue. The ASC command must report `ASC_LAUNCH_OK`.

The final minimal failing pattern is the 24 repeated `BRC_B32` broadcast loads,
arithmetic, and `group=1` compact stores. A single plain grouped byte store and
a plain `ui8→ui16` reinterpret/store both launch successfully; they are not
substitutes for this repro.

## Recorded result

On Ascend950DT device 1 with CANN 9.1.0 and `PTOAS_vmi-v0.1.6`, the VMI probe
printed `PTODSL_COMPILE_OK`. PTOAS emitted `BRC_B32` for the broadcast load but
a dense masked `pto.vsts` for `group=1` (without `ONEPT_B32`); synchronization
then failed with ACL `507035`. The ASC reference built and printed
`ASC_LAUNCH_OK: compact output matched` on the same device/toolchain.
