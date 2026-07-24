# tfjs-native — 機能一覧

[English](../FUNCTIONS.md) | **日本語**

現在提供している機能と今後の予定です。ステータスは [DESIGN.md](./DESIGN.md#12-ロードマップ) のロードマップに対応します。シグネチャは TypeScript です。

## 現在利用可能（M1）

### ランタイム

| API | 説明 |
|---|---|
| `version(): string` | リンクされた libtensorflow のバージョン。 |

### テンソル生成

| API | 説明 |
|---|---|
| `tensor(values, shape?, dtype?): Tensor` | スカラー・矩形のネスト配列・フラットな `TypedArray` から生成。`shape` 省略時は推論。 |
| `scalar(value, dtype?): Tensor` | ランク 0 のテンソル。 |
| `zeros(shape, dtype?): Tensor` | 0 で埋めたテンソル。 |
| `ones(shape, dtype?): Tensor` | 1 で埋めたテンソル。 |

### `Tensor`

| メンバ | 説明 |
|---|---|
| `shape: readonly number[]` | 各次元（`[]` はスカラー）。 |
| `dtype: DTypeName` | 要素の型。 |
| `size: number` | 要素数。 |
| `rank: number` | 次元数。 |
| `isDisposed: boolean` | ネイティブハンドルが解放済みか。 |
| `data(): Promise<ArrayBufferView>` / `dataSync()` | dtype の `TypedArray` として生データを読み出す。 |
| `array(): Promise<NestedList>` / `arraySync()` | `shape` の形のネスト配列として読み出す。 |
| `dispose(): void` | ネイティブハンドルを解放（冪等）。 |

### メモリと dtype

| API | 説明 |
|---|---|
| `tidy(fn)` | `fn` を実行し、戻り値に含まれないテンソルを破棄する。 |
| `DType`, `DEFAULT_DTYPE` | dtype 表。対応: `float32`（既定）, `float64`, `int32`, `int64`（`bigint` で読み出し）, `uint8`, `bool`。`string` / `complex64` は予定。 |

int64 を第一級で扱えます（`BigInt64Array` 経由）。int32 止まりの `@tensorflow/tfjs` との違いです。

### eager op（M2）

eager 実行パスは稼働済みです。**約 1,295 op** を TF の op registry（`TF_GetAllOpList`）から生成しています。tfjs に無い `scatterNd`・`gatherNd`・`nonMaxSuppressionV2`・`conv2D`・`softmax` なども含みます。生成ラッパは入力（`Tensor` / `Tensor[]`）と出力数が型付きで、型・サイズ属性は入力から自動導出し、それ以外は汎用 options で渡します（省略すると op の TF 既定値を使用）。

その上に、より扱いやすいシグネチャの手書き op があります:

| グループ | op |
|---|---|
| 二項 | `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `pow` |
| 単項 | `neg`, `abs`, `exp`, `log`, `sqrt`, `square`, `relu`, `sigmoid`, `tanh` |
| リダクション | `sum`, `mean`, `max`, `min`（`axis`, `keepDims` 対応） |
| 線形代数 | `matMul`（`transposeA` / `transposeB` 対応） |
| 形状 / dtype | `reshape`, `transpose`, `cast` |
| エスケープハッチ | `runOp(name, inputs, attrs, numOutputs)` — 任意の op を名前で実行 |

### SavedModel（M3）

TensorFlow の SavedModel をロードし、signature 経由で推論を実行します。feed/fetch は signature 自身のキーで指定するので、生のグラフテンソル名を扱う必要はありません。

| API | 説明 |
|---|---|
| `loadSavedModel(dir, { tags? }): SavedModel` | `saved_model.pb` を含むディレクトリをロード。tags の既定は `["serve"]`。 |
| `model.signatureNames: string[]` | 利用可能な signature（例: `["serving_default"]`）。 |
| `model.signatures` | signature ごとの入出力キーと、対応するグラフテンソル名。 |
| `model.predict(feeds)` | `serving_default` signature を実行。 |
| `model.run(signature, feeds)` | 名前を指定して signature を実行。 |
| `model.dispose()` | セッションを閉じ、グラフを解放（冪等）。 |

```ts
import { loadSavedModel, tensor } from "tfjs-native";

const model = loadSavedModel("./my_saved_model");
const { y } = model.predict({ x: tensor([1, 2, 3]) });
console.log(await y.array());
model.dispose();
```

### 学習（M4）

自動微分と optimizer。勾配は記録＋リプレイ方式: 損失を eager で実行しながらテープに記録し、グラフに再構築して TensorFlow 自身の勾配定義を適用します（2.10 では 140 op に勾配あり）。

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

## 今後の予定

| マイルストーン | 機能 |
|---|---|
| **M5** | OS 別 prebuild、libtensorflow の自動取得、trusted publishing リリース。 |

## 対象外

libtensorflow C API に露出しないため意図的に除外しています: Keras の高レベル層 API 全体、`tf.data` パイプライン、`tf.function`（autograph）、distribution strategy、ブラウザ / WASM / WebGL バックエンド。詳細は [DESIGN.md](./DESIGN.md) のスコープ節を参照してください。
