# cpptysor

`cpptysor` is a C++17 compiler/runtime prototype for a small tensor language.
The project is shaped around an explicit compiler pipeline:

```text
lexer -> parser -> semantic_analyzer -> frontend_ir -> graph_ir -> execution_plan -> runtime/backend
```

The current focus is making the middle of the compiler real: `graph_ir`
represents tensor dataflow and shape facts, while `execution_plan` turns that
graph into backend-specific scheduling steps for local CPU execution and early
Metal execution.

## Current Status

Implemented today:

- Lexer, parser, semantic analyzer, and frontend IR lowering.
- **Deterministic 32-bit `NodeId` AST Indexing**: Every parsed AST `Expr` and `Stmt` is assigned a unique `NodeId` for precise, span-independent tracking.
- **$O(1)$ Side-Table Lookups**: `SemanticInfo` side-tables use hash maps to resolve expression types, identifiers, calls, declarations, and assignments in constant time.
- **Modular IR Pipeline**: `FrontendLowerer` and `GraphBuilder` are fully modularized with dedicated expression and statement handlers.
- **Unified Layer Constructors**: Unified function and layer constructor calls under `FeCallExpr` carrying `FeTypeKind::Callable`.
- **Built-in `print(...)` Output**: Full support for printing tensors, scalars (`int`, `float`, `bool`), and string literals (`str`).
- Graph IR for straight-line tensor functions/layers.
- Symbolic and known tensor shape metadata in Graph IR.
- Explicit Graph IR parameter values for `linear` weights and bias.
- Execution plans with host/device placement, operation steps, outputs, and trainable parameter metadata.
- Local runtime execution for tensor programs.
- Experimental backward and SGD training paths for supported programs.
- Runtime tensor alignment, metadata-only reshape/flatten views, and workspace buffer reuse.
- Local `matmul + relu` fusion when intermediate preservation is disabled.
- Metal backend plumbing and preflight/capability checks for supported kernels.

Important limits:

- The codebase intentionally uses C++17. Do not introduce C++20-only APIs.
- Graph IR currently supports straight-line graphable functions. Parsed control
  flow exists, but non-straight-line functions are skipped during graph build.
- Runtime tensors are float-backed; dtype is mostly compiler/runtime metadata.
- Input tensors still need CLI `--shape` entries at runtime.
- Training uses synthetic inputs/parameters and a small SGD path, not real data
  loading or serialized model state yet.
- `examples/tensor.ty` is currently an architectural sketch, not a guaranteed
  runnable quick-start program.
- PyTorch, CUDA, and ROCm are represented in backend enums/capability plumbing,
  but codegen/runtime execution is not ported yet.

## Repository Layout

```text
frontend/          lexer, parser, semantic analyzer, builtin op registry
ir/                frontend IR, graph IR, execution plan lowering/validation
runtime/           local tensor runtime, graph executor, backward, train loop
backend/metal/     Metal executor bridge and kernel dispatch integration
cli/               command-line parsing, compiler pipeline driver
tests/             smoke tests for each compiler/runtime stage
bench/             runtime microbenchmark executable
examples/          experimental language sketches
core/              diagnostics and shared source utilities
```

## Build

Requirements:

- Meson
- Ninja
- A C++17 compiler
- On Apple builds, a `metal_bridge.mm` file must exist at one of the paths
  checked by `meson.build`: `backend/metal/metal_bridge.mm`,
  `../native/metal_bridge.mm`, or `../tysor/native/metal_bridge.mm`.

Build:

```bash
meson setup build --buildtype=debug
meson compile -C build
```

Run the full test suite:

```bash
meson test -C build
```

Run the runtime benchmark:

```bash
./build/cpptysor_runtime_bench
```

## CLI

Show help:

```bash
./build/tysor --help
```

Common inspection flow:

```bash
./build/tysor path/to/program.ty --tokens --ast --semantics --ir --graph --plan
```

Run a local program:

```bash
./build/tysor path/to/program.ty --run --shape x=2x3
```

Run a training config:

```bash
./build/tysor path/to/program.ty --train --shape x=2x3 --shape target=2x2
```

Select a backend:

```bash
./build/tysor path/to/program.ty --plan --backend local
./build/tysor path/to/program.ty --plan --backend metal
```

Probe Metal availability:

```bash
./build/tysor --metal-device
```

## Minimal Runnable Program

Create a small `.ty` file:

```tysor
layer model(x: tensor[float32]): tensor[float32]:
  let proj = linear(3, 2, true)
  return proj(x) -> Tanh()
```

Inspect the compiler output:

```bash
./build/tysor quickstart.ty --graph --plan
```

Run it with a synthetic input tensor:

```bash
./build/tysor quickstart.ty --run --shape x=2x3
```

## Minimal Training Program

```tysor
layer model(x: tensor[float32], target: tensor[float32]): tensor[float32]:
  let proj = linear(3, 2, true)
  let logits = proj(x)
  let loss = cross_entropy(logits, target)
  return loss

config model:
  optimizer = "sgd"
  lr = 0.05
  iteration = 2
  objective = loss
```

Run:

```bash
./build/tysor train.ty --train --shape x=2x3 --shape target=2x2
```

## Language Sketch

Top-level forms:

```tysor
config settings:
  hidden: int32 = 128

fn helper(x: tensor[float32]): tensor[float32]:
  return sqrt(x)

layer model(x: tensor[float32]): tensor[float32]:
  return helper(x)
```

Useful syntax:

- `let x = expr`
- `let mut x = expr`
- `x = expr` for mutable bindings
- `return expr`
- `x -> relu()` pipeline syntax
- `x -> relu()[2]` repeated pipeline stage
- `proj(x)` callable application
- `x -> proj()` pipeline callable application
- tensor annotations such as `tensor[float32]` and
  `tensor[float32, [batch, 3]]`

Supported builtin families:

- Trainable/callable constructors: `linear`, `Embedding`
- Activations/callables: `SiLU`, `GELU`, `Tanh`, `Sigmoid`, `Softmax`,
  `Dropout`, `RMSNorm`
- Primitive tensor ops: `matmul`, `relu`, `scale`, `print`
- Library tensor ops: `rms_norm`, `cross_entropy`, `rope`, `reshape`,
  `transpose`, `sum`, `mean`, `sqrt`, `rsqrt`, `causal_mask`,
  `flatten_heads`, `repeat_kv`

Not every builtin has identical support across local execution, backward,
training, and Metal execution. The execution plan capability checks are the
source of truth for a backend.

## Compiler Design Notes

`semantic_analyzer` owns language-level correctness:

- symbol resolution
- mutability checks
- builtin arity/type checks
- config field validation
- training config validation
- side-table indexing with deterministic `NodeId` map resolution

`graph_ir` owns tensor graph correctness:

- dense value ids
- producer/consumer ordering
- tensor shape metadata
- symbolic dimensions such as `batch`
- explicit model parameters for trainable ops
- straight-line dataflow validation

`execution_plan` owns execution decisions:

- value placement (`Host` or `Device`)
- op lowering into backend steps
- upload/dispatch/download steps for device backends
- backend capability validation
- local graph optimizations such as `matmul_relu`

`runtime` owns actual execution:

- aligned tensor storage
- copy-on-write metadata views for reshape/flatten
- local tensor kernels
- workspace pooling for temporary tensors
- backward/training support for the current supported subset

## Development Commands

Build everything:

```bash
meson compile -C build
```

Run one smoke executable:

```bash
./build/cpptysor_graph_ir_smoke
./build/cpptysor_execution_plan_smoke
./build/cpptysor_graph_executor_smoke
./build/cpptysor_train_smoke
```

Run all tests:

```bash
meson test -C build
```

Check formatting-related whitespace before committing:

```bash
git diff --check
```

## Near-Term Roadmap

High-value next steps:

- Implement unified declarative `OpRegistry` for centralized built-in op type inference and kernel dispatch.
- Lower remaining trainable callables, especially `Embedding`, into explicit graph parameter dataflow.
- Add real input/data loading instead of synthetic runtime tensors.
- Persist model parameter state for training.
- Expand Graph IR autodiff instead of relying on executor-specific backward paths.
- Grow backend kernels and capability tests for Metal/CUDA/ROCm.
- Move more tensor shape facts out of CLI runtime shape hints and into compiler inference where possible.
