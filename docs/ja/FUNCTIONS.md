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

## 今後の予定

| マイルストーン | 機能 |
|---|---|
| **M3** | SavedModel / GraphDef のロードと推論（`loadSavedModel`, `model.run`）。 |
| **M4** | 薄い学習レイヤ: GradientTape 相当、`sgd`/`adam`/`rmsprop`、`minimize`。 |
| **M5** | OS 別 prebuild、libtensorflow の自動取得、trusted publishing リリース。 |

## 対象外

libtensorflow C API に露出しないため意図的に除外しています: Keras の高レベル層 API 全体、`tf.data` パイプライン、`tf.function`（autograph）、distribution strategy、ブラウザ / WASM / WebGL バックエンド。詳細は [DESIGN.md](./DESIGN.md) のスコープ節を参照してください。
