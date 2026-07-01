TileLang DSL examples live here.

Examples in this subtree should import `tilelang_dsl` as their package
entrypoint once the package wiring is added.

Current examples:
- `v1_emit_mlir_demo.py`: minimal descriptor/materialization demo
- `v1_elementwise_tail_demo.py`: guide-aligned elementwise authoring demo that
  covers DMA, explicit `strict_vecscope`, dynamic loop bound, and typed tail
  mask lowering
- `v1_template_slot_multiop_demo.py`: shared kernel-body demo for
  `tadd`/`tsub`/`tmul`/`tdiv` using `ops=[...]`, `templates={...}`, and
  `pto.tpl("core", ...)`
- `v1_tadd_implicit_vecscope_demo.py`: advanced-mode flattened `TADD` example
  with implicit `pto.vecscope` inference, dynamic Tile `valid_shape`, generic
  dtype selection, partial-dynamic `valid_shape` modes, and `vlds`/`vsts`
  tile indexing sugar
- `v1_tbinop_2d_nopostupdate_demo.py`: a representative TileLang DSL v1
  expansion of `pto::TBinOps_2D_NoPostUpdate` using `vadd`
- `v1_verify_smoke.py`: minimal verify smoke that is expected to pass the repo
  `ptoas --pto-backend=vpto` legality path

Typical usage from the repository root:

```bash
python3 tilelang-dsl/examples/v1_emit_mlir_demo.py
python3 tilelang-dsl/examples/v1_emit_mlir_demo.py /tmp/tilelang_demo.mlir
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_elementwise_tail_demo.py
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_template_slot_multiop_demo.py
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_template_slot_multiop_demo.py tsub
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_template_slot_multiop_demo.py tmul f16
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_tadd_implicit_vecscope_demo.py
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_tadd_implicit_vecscope_demo.py f16
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_tadd_implicit_vecscope_demo.py f16 rows
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_tadd_implicit_vecscope_demo.py f16 cols
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_tbinop_2d_nopostupdate_demo.py
PYTHONPATH=$PWD/tilelang-dsl/python python3 tilelang-dsl/examples/v1_verify_smoke.py
```
