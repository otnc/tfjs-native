# tfjs-native — Functions & Features

**English** | [日本語](./ja/FUNCTIONS.md)

What the package exposes today and what is planned. Status reflects the milestone roadmap in [DESIGN.md](./DESIGN.md#12-roadmap). Signatures use TypeScript.

## Available now (M1)

### Runtime

| API | Description |
|---|---|
| `version(): string` | Linked libtensorflow version. |

### Tensor creation

| API | Description |
|---|---|
| `tensor(values, shape?, dtype?): Tensor` | From a scalar, a rectangular nested array, or a flat `TypedArray`. Shape inferred when omitted. |
| `scalar(value, dtype?): Tensor` | Rank-0 tensor. |
| `zeros(shape, dtype?): Tensor` | Tensor filled with 0. |
| `ones(shape, dtype?): Tensor` | Tensor filled with 1. |

### `Tensor`

| Member | Description |
|---|---|
| `shape: readonly number[]` | Dimensions (`[]` = scalar). |
| `dtype: DTypeName` | Element type. |
| `size: number` | Element count. |
| `rank: number` | Number of dimensions. |
| `isDisposed: boolean` | Whether the native handle was freed. |
| `data(): Promise<ArrayBufferView>` / `dataSync()` | Read the raw data as the dtype's `TypedArray`. |
| `array(): Promise<NestedList>` / `arraySync()` | Read back as a nested JS array shaped like `shape`. |
| `dispose(): void` | Free the native handle (idempotent). |

### Memory & dtypes

| API | Description |
|---|---|
| `tidy(fn)` | Runs `fn`, disposing tensors it created except those in its return value. |
| `DType`, `DEFAULT_DTYPE` | dtype table. Supported: `float32` (default), `float64`, `int32`, `int64` (read back as `bigint`), `uint8`, `bool`. `string` / `complex64` are planned. |

int64 is first-class here (via `BigInt64Array`), unlike `@tensorflow/tfjs`, which stops at int32.

### Eager ops (M2)

The eager execution path is live. **~1,295 ops are generated** from the TF op registry (`TF_GetAllOpList`) — including ops tfjs lacks, such as `scatterNd`, `gatherNd`, `nonMaxSuppressionV2`, `conv2D`, `softmax`. Generated wrappers have typed inputs (`Tensor` / `Tensor[]`) and a typed output count; type/size attributes are auto-derived from inputs, and the rest are passed via a generic options object (omit one to use the op's TF default).

A set of hand-written ops sits on top with friendlier signatures:

| Group | Ops |
|---|---|
| Binary | `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `pow` |
| Unary | `neg`, `abs`, `exp`, `log`, `sqrt`, `square`, `relu`, `sigmoid`, `tanh` |
| Reduction | `sum`, `mean`, `max`, `min` (with `axis`, `keepDims`) |
| Linear algebra | `matMul` (with `transposeA` / `transposeB`) |
| Shape / dtype | `reshape`, `transpose`, `cast` |
| Escape hatch | `runOp(name, inputs, attrs, numOutputs)` — call any op by name |

## Planned

| Milestone | Feature |
|---|---|
| **M3** | SavedModel / GraphDef load and inference (`loadSavedModel`, `model.run`). |
| **M4** | Thin training layer: GradientTape equivalent, `sgd`/`adam`/`rmsprop`, `minimize`. |
| **M5** | Prebuilt binaries per OS, automatic libtensorflow fetch, trusted-publish releases. |

## Out of scope

Not exposed by the libtensorflow C API, so intentionally excluded: the full Keras high-level layer API, `tf.data` pipelines, `tf.function` (autograph), distribution strategies, and any browser / WASM / WebGL backend. See the Scope section in [DESIGN.md](./DESIGN.md).
