## Quick Start

**Note on mask pattern enums**: For brevity, examples in this guide use `PAT` as an alias for `pto.MaskPattern` (e.g., `PAT.ALL` instead of `pto.MaskPattern.PAT_ALL`). You can create this alias with `from pto import MaskPattern as PAT` or `PAT = pto.MaskPattern`.

TileLang DSL provides the following core constructs for kernel authoring:

- `TensorView` – Access global memory (GM) tensors
- `Tile` – Local computation buffers in unified buffer (UB)
- Base vector operations (`make_mask`, `vlds`, `vmuls`, `vadd`, `vsts`) – Perform vector computations

A typical kernel follows the GM → UB → vector compute → GM pattern:

```python
import tilelang_dsl as pto

@pto.vkernel(target="a5", op="scale", dtypes=[(pto.f32, pto.f32, pto.f32, pto.f32)])
def tile_scale(
    input_tensor: pto.TensorView,
    output_tensor: pto.TensorView,
    work_tile: pto.Tile,
    scale_factor: pto.f32,
):
    dim0 = 4
    dim1 = 16

    # Stage one GM tile into UB.
    # GM -> UB data movement (implementation detail)

    # Run vector compute over the UB tile using tile indexing sugar.
    for i in range(0, dim0):
        mask = pto.make_mask(pto.f32, PAT.ALL)
        vec = pto.vlds(work_tile[i, 0:])
        scaled = pto.vmuls(vec, scale_factor, mask)
        pto.vsts(scaled, work_tile[i, 0:], mask)

    # Write the UB result back to GM.
    # UB -> GM data movement (implementation detail)
```

The example illustrates the key components of a TileLang kernel:

1. **`TensorView` parameters** – Access global memory tensors
2. **`Tile` parameters** – Local computation buffers in unified buffer (UB)
3. **Base vector operations** (`make_mask`, `vlds`, `vmuls`, `vadd`, `vsts`) – Perform vector computations

Here is a second example with two inputs and one output:

```python
@pto.vkernel(
    target="a5",
    op="elementwise_add",
    dtypes=[(pto.f32, pto.f32, pto.f32, pto.f32, pto.f32, pto.f32)],
)
def elementwise_add(
    lhs_gm: pto.TensorView,
    rhs_gm: pto.TensorView,
    out_gm: pto.TensorView,
    lhs_tile: pto.Tile,
    rhs_tile: pto.Tile,
    dst_tile: pto.Tile,
):
    dim0 = 4
    dim1 = 16

    # GM -> UB data movement (implementation detail)

    for lane in range(0, 256, 64):
        mask = pto.make_mask(pto.f32, PAT.ALL)
        lhs_vec = pto.vlds(lhs_tile, lane)
        rhs_vec = pto.vlds(rhs_tile, lane)
        summed = pto.vadd(lhs_vec, rhs_vec, mask)
        pto.vsts(summed, dst_tile, lane, mask)

    # UB -> GM data movement (implementation detail)
```

Both examples follow the same fundamental pattern: load data from global memory into local tiles, perform vector operations, and store results back. The compiler automatically infers vector-scope boundaries for the base vector operations. The `Tile` parameters are specialized to concrete shapes during compilation. Later sections cover advanced features such as matchers, template slots, raw pointer operations, and explicit scope management with `strict_vecscope`.

