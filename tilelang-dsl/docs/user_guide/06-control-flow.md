## Control Flow

### Vector Scopes

The TileLang DSL supports implicit vector scope inference, allowing developers to write vector operations directly without explicit `pto.vecscope()` blocks. The compiler automatically groups consecutive, data-dependent vector operations into implicit vector scopes during lowering.

#### Implicit Scope Inference

**Note:** `pto.vecscope()` is supported. Automatic scope inference runs only when the kernel does **not** contain explicit `with pto.vecscope():` blocks.

When you write vector operations like `pto.vlds`, `pto.vadd`, `pto.vsts` directly in your code, the compiler's **Scope Inference Pass** analyzes the control flow graph and automatically creates vector scopes:

```python
# No explicit vecscope needed - compiler infers scope boundaries
vec = pto.vlds(outer_ptr, offset)
result = pto.vadd(vec, vec, all_mask)
pto.vsts(result, dst_ptr, offset, all_mask)
```

The compiler automatically groups these three operations into a single implicit vector scope because they form a data-dependent chain (when no explicit `pto.vecscope()` appears in the kernel).

**Scope boundary rules:**
1. **Control flow boundaries**: Branches (`if`/`else`), loops (`for`/`while`), and function calls create implicit scope boundaries
2. **Scalar operations**: Non-vector operations (e.g., scalar arithmetic, pointer arithmetic) create boundaries
3. **Explicit scope blocks**: User-defined `vecscope` and `strict_vecscope` blocks create hard boundaries

#### Explicit Scope Boundaries with `strict_vecscope` [Advanced Tier]

##### `pto.strict_vecscope(*captures: AnyType) -> ContextManager[Tuple[AnyType, ...]]`

**Description**: Creates an explicit vector scope boundary with explicit value captures. Values used inside the scope must be passed as arguments; implicit capture from outer scope is rejected.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `*captures` | `AnyType` | Variable number of values to be captured and passed into the scope |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `context_manager` | `ContextManager[Tuple[AnyType, ...]]` | Context manager that yields a tuple of captured values when entered |

**Constraints**:
- The scope body cannot implicitly capture values from the surrounding scope; all used values must be passed as `captures`.
- Creates a hard boundary that prevents the compiler from merging vector operations across the scope boundary.
- Useful for performance optimization, debugging, resource management, and hardware compatibility.

For precise control over scope boundaries, use explicit `strict_vecscope` blocks. These create hard boundaries that prevent the compiler from merging operations across the block boundary:

```python
with pto.strict_vecscope(src_ptr, dst_ptr, start, end) as (s, d, lb, ub):
    # Operations inside this block are isolated from outside
    # Compiler will not merge operations across this boundary
    for i in range(lb, ub, 64):
        vec = pto.vlds(s, i)
        pto.vsts(vec, d, i, all_mask)
```

**Use cases for strict_vecscope:**
- Performance optimization: Isolate critical vector computation regions
- Debugging: Create explicit boundaries to isolate vector operations
- Resource management: Control vector register allocation boundaries
- Compatibility: Ensure deterministic scope placement for hardware constraints

#### Explicit Scope Blocks with `vecscope`

`pto.vecscope` provides an explicit vector-scope boundary without strict capture ABI constraints:

```python
with pto.vecscope():
    vec = pto.vlds(src, 0)
    vec = pto.vadd(vec, vec, mask)
    pto.vsts(vec, dst, 0, mask)
```

**Rules**:
- `pto.vecscope()` takes no positional/keyword arguments.
- `pto.vecscope()` does not support `as (...)` bindings.
- When any explicit `pto.vecscope()` is present in a kernel body, automatic vecscope inference is disabled for that kernel.

### Inline Procedures (`@pto.inline_proc`)

TileLang DSL supports reusable top-level procedures decorated with `@pto.inline_proc`.
`inline_proc` follows function-call semantics in frontend IR and is force-inlined
later by the VPTO backend mainline in `ptoas`.

```python
@pto.inline_proc
def store_row(dst: pto.Tile, src: pto.Tile, row: pto.i32):
    vec = pto.vlds(src[row, 0:])
    mask = pto.make_mask(dst.element_type, pto.PAT.ALL)
    pto.vsts(vec, dst[row, 0:], mask)
    return None

@pto.vkernel(op="pto.row_copy", dtypes=[(pto.f32, pto.f32, pto.i32)])
def row_copy(dst: pto.Tile, src: pto.Tile, row: pto.i32):
    store_row(dst, src, row)
    return None
```

Important semantics:

- `pto.<surface>(...)` and bare helper calls are different mechanisms.
- Calls written as `pto.vadd(...)`, `pto.vdiv(...)`, `pto.vlds(...)`, etc. target
  built-in TileLang/VPTO surfaces directly.
- Calls written as bare Python names such as `store_row(...)` target a
  user-defined `@pto.inline_proc` helper when the callee name resolves to a
  registered top-level inline procedure in the current module.
- `inline_proc` helpers do not live in the `pto` namespace; using the same
  basename as a `pto.<surface>` op is allowed because the frontend distinguishes
  `pto.xxx(...)` from bare `xxx(...)` calls.
- Frontend preserves helper `func.func` and `func.call` in `mlir_text()` output.
- VPTO backend mainline force-inlines helper calls before downstream lowering.
- Helper definitions support default parameter values.
- Helper calls support positional arguments and keyword arguments.
- Helper calls can appear in statement and expression positions.
- Helper definitions can use trailing `return <expr>` to return values.
- Implicit capture is rejected except module-level globals whose current bound value is `bool`/`int`/`float`/`str`; pass other required values as explicit arguments.
- Recursive/mutually-recursive helper call graphs are rejected.
- `*args`, `**kwargs`, and keyword-only parameters are unsupported in current version.

Shared helpers can live in a separate Python file in the template directory and
be imported directly by templates:

```python
# shared_rows.py
import tilelang_dsl as pto

@pto.inline_proc
def touch_row(dst: pto.Tile, row: pto.i32):
    mask = pto.make_mask(dst.element_type, pto.PAT.ALL)
    vec = pto.vlds(dst[row, 0:])
    pto.vsts(vec, dst[row, 0:], mask)
    return None

# trow_template.py
import tilelang_dsl as pto
from shared_rows import touch_row

@pto.vkernel(op="pto.row_touch", dtypes=[(pto.f32, pto.i32)])
def row_touch(dst: pto.Tile, row: pto.i32):
    touch_row(dst, row)
    return None
```

Only directly imported `@pto.inline_proc` helpers are part of this shared-helper
surface. Ordinary Python functions remain unsupported in DSL bodies, and
qualified calls such as `shared_rows.touch_row(...)` are not part of this
version. If multiple imported helpers expose the same bare name, the frontend
rejects the template instead of choosing one by import order.

### Loops

Counted loops use Python's `range` syntax:

```python
for i in range(lb, ub, step):
    # Loop body
    mask, rem = pto.make_mask(pto.f32, remaining)
    # ...
```

Loop-carried state is automatically handled through variable updates within the loop.

### Conditionals

`if` statements support value merging:

```python
flag: pto.i1 = some_condition
step: pto.i32 = 0

if flag:
    step = pto.i32(64)
else:
    step = pto.i32(128)

# 'step' here is the merged result from both branches
```

Variables defined in only one branch are local to that branch. Variables assigned
in both branches with the same name and type are merged even when the binding is
first introduced inside the `if`/`else`.
