# tfjs-native — Functions & Features

**English** | [日本語](./ja/FUNCTIONS.md)

What the package exposes today, organized by the milestone that introduced it (see the [roadmap](./DESIGN.md#12-roadmap)). Signatures use TypeScript.

## Tensors (M1)

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

## Eager ops (M2)

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

## SavedModel (M3)

Load a TensorFlow SavedModel and run inference through its signatures. Feeds and fetches are keyed by the signature's own names, so you never touch raw graph tensor names.

| API | Description |
|---|---|
| `loadSavedModel(dir, { tags? }): SavedModel` | Loads the directory containing `saved_model.pb`. Tags default to `["serve"]`. |
| `model.signatureNames: string[]` | Available signatures (e.g. `["serving_default"]`). |
| `model.signatures` | Per-signature input/output keys and their graph tensor names. |
| `model.predict(feeds)` | Runs the `serving_default` signature. |
| `model.run(signature, feeds)` | Runs a named signature. |
| `model.dispose()` | Closes the session and frees the graph (idempotent). |

```ts
import { loadSavedModel, tensor } from "tfjs-native";

const model = loadSavedModel("./my_saved_model");
const { y } = model.predict({ x: tensor([1, 2, 3]) });
console.log(await y.array());
model.dispose();
```

## Training (M4)

Automatic differentiation and optimizers. Gradients use record-and-replay: the loss runs eagerly while a tape records it, then it is rebuilt as a graph so TensorFlow's own gradient definitions apply (140 ops have gradients in 2.10).

| API | Description |
|---|---|
| `grads(fn, xs): Tensor[]` | Gradients of `fn()`'s result w.r.t. the tensors `xs`. |
| `valueAndGrads(fn, xs)` | The same, plus `fn`'s value. |
| `variable(initial, name?): Variable` | A trainable, mutable tensor holder (`.value`, `.assign`, `.dispose`). |
| `sgd({ learningRate, momentum? })` | SGD optimizer. |
| `adam({ learningRate?, beta1?, beta2?, epsilon? })` | Adam optimizer. |
| `rmsprop({ learningRate?, decay?, epsilon? })` | RMSProp optimizer. |
| `optimizer.minimize(fn, vars): Tensor` | One training step; returns the loss. Caches the loss graph across steps. |

```ts
import { variable, sgd, tensor, mul, add, sub, mean, square } from "tfjs-native";

const w = variable(tensor([0]));
const b = variable(tensor([0]));
const xs = tensor([0, 1, 2, 3]);
const ys = tensor([2, 5, 8, 11]); // y = 3x + 2
const opt = sgd({ learningRate: 0.05 });

for (let i = 0; i < 400; i++) {
  opt.minimize(() => mean(square(sub(add(mul(w.value, xs), b.value), ys))), [w, b]).dispose();
}
// w ≈ 3, b ≈ 2
```

A requested gradient that cannot exist is reported, not silently zeroed: an input that does not affect the result throws, and a path through a non-differentiable op (e.g. `floor`, `equal`) throws too.

## Out of scope

Not exposed by the libtensorflow C API, so intentionally excluded: the full Keras high-level layer API, `tf.data` pipelines, `tf.function` (autograph), distribution strategies, and any browser / WASM / WebGL backend. See the Scope section in [DESIGN.md](./DESIGN.md).
