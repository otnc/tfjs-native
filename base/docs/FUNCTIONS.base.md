:::kiritan{locale=en}
# tfjs-native — Functions & Features
:::

:::kiritan{locale=ja}
# tfjs-native — 機能一覧
:::

::kiritan{switcher}

:::kiritan{locale=en}
What the package exposes today, organized by the milestone that introduced it (see the [roadmap](./DESIGN.md#12-roadmap)). Signatures use TypeScript.
:::

:::kiritan{locale=ja}
現在提供している機能を、導入されたマイルストーン別にまとめています（[ロードマップ](./DESIGN.md#12-ロードマップ)参照）。シグネチャは TypeScript です。
:::

:::kiritan{locale=en}
## Tensors (M1)
:::

:::kiritan{locale=ja}
## テンソル（M1）
:::

:::kiritan{locale=en}
### Runtime

| API | Description |
|---|---|
| `version(): string` | Linked libtensorflow version. |
:::

:::kiritan{locale=ja}
### ランタイム

| API | 説明 |
|---|---|
| `version(): string` | リンクされた libtensorflow のバージョン。 |
:::

:::kiritan{locale=en}
### Tensor creation

| API | Description |
|---|---|
| `tensor(values, shape?, dtype?): Tensor` | From a scalar, a rectangular nested array, or a flat `TypedArray`. Shape inferred when omitted. Without a dtype, string or `Uint8Array` leaves give `string`, anything else `float32`. |
| `scalar(value, dtype?): Tensor` | Rank-0 tensor. |
| `zeros(shape, dtype?): Tensor` | Tensor filled with 0 (empty strings for `string`). |
| `ones(shape, dtype?): Tensor` | Tensor filled with 1 (`1 + 0i` for `complex64`; not defined for `string`). |
:::

:::kiritan{locale=ja}
### テンソル生成

| API | 説明 |
|---|---|
| `tensor(values, shape?, dtype?): Tensor` | スカラー・矩形のネスト配列・フラットな `TypedArray` から生成。`shape` 省略時は推論。`dtype` を省略すると、要素が文字列か `Uint8Array` なら `string`、それ以外は `float32` になる。 |
| `scalar(value, dtype?): Tensor` | ランク 0 のテンソル。 |
| `zeros(shape, dtype?): Tensor` | 0 で埋めたテンソル(`string` では空文字列)。 |
| `ones(shape, dtype?): Tensor` | 1 で埋めたテンソル(`complex64` では `1 + 0i`。`string` では使えない)。 |
:::

### `Tensor`

:::kiritan{locale=en}
| Member | Description |
|---|---|
| `shape: readonly number[]` | Dimensions (`[]` = scalar). |
| `dtype: DTypeName` | Element type. |
| `size: number` | Element count. |
| `rank: number` | Number of dimensions. |
| `isDisposed: boolean` | Whether the native handle was freed. |
| `data(): Promise<ArrayBufferView>` / `dataSync()` | Read the raw data as the dtype's `TypedArray`. Throws for `string`. |
| `bytes(): Promise<Uint8Array[]>` / `bytesSync()` | Read a `string` tensor's elements as raw bytes, flat in row-major order. |
| `array(): Promise<NestedList>` / `arraySync()` | Read back as a nested JS array shaped like `shape`. `string` elements are decoded as UTF-8. |
| `dispose(): void` | Free the native handle (idempotent). |
:::

:::kiritan{locale=ja}
| メンバ | 説明 |
|---|---|
| `shape: readonly number[]` | 各次元（`[]` はスカラー）。 |
| `dtype: DTypeName` | 要素の型。 |
| `size: number` | 要素数。 |
| `rank: number` | 次元数。 |
| `isDisposed: boolean` | ネイティブハンドルが解放済みか。 |
| `data(): Promise<ArrayBufferView>` / `dataSync()` | dtype の `TypedArray` として生データを読み出す。`string` では例外を投げる。 |
| `bytes(): Promise<Uint8Array[]>` / `bytesSync()` | `string` テンソルの各要素をバイト列として、行優先のフラットな配列で読み出す。 |
| `array(): Promise<NestedList>` / `arraySync()` | `shape` の形のネスト配列として読み出す。`string` の要素は UTF-8 としてデコードする。 |
| `dispose(): void` | ネイティブハンドルを解放（冪等）。 |
:::

:::kiritan{locale=en}
### Memory & dtypes

| API | Description |
|---|---|
| `tidy(fn)` | Runs `fn`, disposing tensors it created except those in its return value. |
| `DType`, `DEFAULT_DTYPE` | dtype table. Supported: `float32` (default), `float64`, `int32`, `int64` (read back as `bigint`), `uint8`, `bool`, `string`, `complex64`. |

`string` tensors hold UTF-8 text or arbitrary bytes (for example an encoded image to feed `decodeJpeg`). Pass `Uint8Array` leaves for binary data and read it back with `bytes()`, since `array()` decodes as UTF-8.

`complex64` follows tfjs: build one with `complex(real, imag)` (or from an interleaved `[re, im, ...]` `Float32Array`), take it apart with `real` / `imag`, and expect `dataSync()` and the innermost dimension of `array()` to hold interleaved pairs.

int64 is first-class here (via `BigInt64Array`), unlike `@tensorflow/tfjs`, which stops at int32.
:::

:::kiritan{locale=ja}
### メモリと dtype

| API | 説明 |
|---|---|
| `tidy(fn)` | `fn` を実行し、戻り値に含まれないテンソルを破棄する。 |
| `DType`, `DEFAULT_DTYPE` | dtype 表。対応: `float32`（既定）, `float64`, `int32`, `int64`（`bigint` で読み出し）, `uint8`, `bool`, `string`, `complex64`。 |

`string` テンソルには UTF-8 のテキストも任意のバイト列(`decodeJpeg` に渡すエンコード済み画像など)も入れられます。`array()` は UTF-8 としてデコードするので、バイナリは `Uint8Array` の要素で渡し、`bytes()` で読み出してください。

`complex64` は tfjs と同じ扱いです。`complex(real, imag)`(または `[re, im, ...]` と交互に並べた `Float32Array`)で作り、`real` / `imag` で分解します。`dataSync()` と `array()` の最内次元には実部と虚部が交互に入ります。

int64 を第一級で扱えます（`BigInt64Array` 経由）。int32 止まりの `@tensorflow/tfjs` との違いです。
:::

:::kiritan{locale=en}
## Eager ops (M2)

The eager execution path is live. **~1,334 ops are generated** from the TF op registry (`TF_GetAllOpList`) — including ops tfjs lacks, such as `scatterNd`, `gatherNd`, `nonMaxSuppressionV2`, `conv2D`, `softmax`. Generated wrappers have typed inputs (`Tensor` / `Tensor[]`) and a typed output count; type/size attributes are auto-derived from inputs, and the rest are passed via a generic options object (omit one to use the op's TF default).

A set of hand-written ops sits on top with friendlier signatures:

| Group | Ops |
|---|---|
| Binary | `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `pow` |
| Unary | `neg`, `abs`, `exp`, `log`, `sqrt`, `square`, `relu`, `sigmoid`, `tanh` |
| Reduction | `sum`, `mean`, `max`, `min` (with `axis`, `keepDims`) |
| Linear algebra | `matMul` (with `transposeA` / `transposeB`) |
| Shape / dtype | `reshape`, `transpose`, `cast` |
| Escape hatch | `runOp(name, inputs, attrs, numOutputs)` — call any op by name |
:::

:::kiritan{locale=ja}
## eager op（M2）

eager 実行パスは稼働済みです。**約 1,334 op** を TF の op registry（`TF_GetAllOpList`）から生成しています。tfjs に無い `scatterNd`・`gatherNd`・`nonMaxSuppressionV2`・`conv2D`・`softmax` なども含みます。生成ラッパは入力（`Tensor` / `Tensor[]`）と出力数が型付きで、型・サイズ属性は入力から自動導出し、それ以外は汎用 options で渡します（省略すると op の TF 既定値を使用）。

その上に、より扱いやすいシグネチャの手書き op があります:

| グループ | op |
|---|---|
| 二項 | `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `pow` |
| 単項 | `neg`, `abs`, `exp`, `log`, `sqrt`, `square`, `relu`, `sigmoid`, `tanh` |
| リダクション | `sum`, `mean`, `max`, `min`（`axis`, `keepDims` 対応） |
| 線形代数 | `matMul`（`transposeA` / `transposeB` 対応） |
| 形状 / dtype | `reshape`, `transpose`, `cast` |
| エスケープハッチ | `runOp(name, inputs, attrs, numOutputs)` — 任意の op を名前で実行 |
:::

:::kiritan{locale=en}
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
:::

:::kiritan{locale=ja}
## SavedModel（M3）

TensorFlow の SavedModel をロードし、signature 経由で推論を実行します。feed/fetch は signature 自身のキーで指定するので、生のグラフテンソル名を扱う必要はありません。

| API | 説明 |
|---|---|
| `loadSavedModel(dir, { tags? }): SavedModel` | `saved_model.pb` を含むディレクトリをロード。tags の既定は `["serve"]`。 |
| `model.signatureNames: string[]` | 利用可能な signature（例: `["serving_default"]`）。 |
| `model.signatures` | signature ごとの入出力キーと、対応するグラフテンソル名。 |
| `model.predict(feeds)` | `serving_default` signature を実行。 |
| `model.run(signature, feeds)` | 名前を指定して signature を実行。 |
| `model.dispose()` | セッションを閉じ、グラフを解放（冪等）。 |
:::

```ts
import { loadSavedModel, tensor } from "tfjs-native";

const model = loadSavedModel("./my_saved_model");
const { y } = model.predict({ x: tensor([1, 2, 3]) });
console.log(await y.array());
model.dispose();
```

:::kiritan{locale=en}
## Training (M4)

Automatic differentiation and optimizers. Gradients use record-and-replay: the loss runs eagerly while a tape records it, then it is rebuilt as a graph so TensorFlow's own gradient definitions apply (143 ops have gradients in 2.15).

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
:::

:::kiritan{locale=ja}
## 学習（M4）

自動微分と optimizer。勾配は記録＋リプレイ方式: 損失を eager で実行しながらテープに記録し、グラフに再構築して TensorFlow 自身の勾配定義を適用します(2.15 では 143 op に勾配あり)。

| API | 説明 |
|---|---|
| `grads(fn, xs): Tensor[]` | `fn()` の結果を `xs` で微分した勾配。 |
| `valueAndGrads(fn, xs)` | 上に加えて `fn` の値も返す。 |
| `variable(initial, name?): Variable` | 学習可能な可変テンソルの入れ物（`.value`, `.assign`, `.dispose`）。 |
| `sgd({ learningRate, momentum? })` | SGD optimizer。 |
| `adam({ learningRate?, beta1?, beta2?, epsilon? })` | Adam optimizer。 |
| `rmsprop({ learningRate?, decay?, epsilon? })` | RMSProp optimizer。 |
| `optimizer.minimize(fn, vars): Tensor` | 1 ステップ学習し損失を返す。損失グラフはステップ間でキャッシュ。 |

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

取れない勾配は黙って 0 にせず報告します。結果に影響しない入力は例外になり、微分不能な op（`floor`, `equal` など）を通る経路も例外になります。
:::

:::kiritan{locale=en}
## Out of scope

Not exposed by the libtensorflow C API, so intentionally excluded: the full Keras high-level layer API, `tf.data` pipelines, `tf.function` (autograph), distribution strategies, and any browser / WASM / WebGL backend. See the Scope section in [DESIGN.md](./DESIGN.md).
:::

:::kiritan{locale=ja}
## 対象外

libtensorflow C API に露出しないため意図的に除外しています: Keras の高レベル層 API 全体、`tf.data` パイプライン、`tf.function`（autograph）、distribution strategy、ブラウザ / WASM / WebGL バックエンド。詳細は [DESIGN.md](./DESIGN.md) のスコープ節を参照してください。
:::
