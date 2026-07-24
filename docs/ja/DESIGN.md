# tfjs-native 設計書

[English](../DESIGN.md) | **日本語**

Node.js から、オリジナルの TensorFlow（C/C++）を型付き TypeScript API で呼び出すネイティブ wrapper です。運用規約（コミット・言語・リリース）は [RULES.md](./RULES.md) を参照してください。本書は**設計**を扱います。

---

## 1. 目的とゴール

### 1.1 課題

`@tensorflow/tfjs` の op カバレッジは狭いです。tfjs-converter がマップする TF op は約 **287**（[supported_ops.md](https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/docs/supported_ops.md)）で、オリジナルの TF op registry には **1,400〜2,000+** あります。RNN 系の op はほぼ未マップで、`NonMaxSuppressionV2` / `TensorArray*` / `ScatterNd` / `GatherNd` はモデル変換でよく落ちます。`@tensorflow/tfjs-node` は既に libtensorflow の C API をバインドしていますが、公開しているのは tfjs の op サブセットだけで、**C API の裏にある全 op registry を活かせていません**。

### 1.2 North Star（唯一のゴール）

> libtensorflow C API が提供する**すべての op** を eager 実行で公開し、TypeScript から 100% 呼べるようにする。その上に **SavedModel の実行**と、TS 側で実装する**薄い高レベル学習 API（勾配 / optimizer）**を載せる。

「100%」の意味は **「C API が露出する範囲の 100%」**。この境界は動かしません。

### 1.3 スコープ

**含む**

- すべての op の eager 実行（`TFE_*`）。op 一覧は `TF_GetAllOpList()` から取得しコード生成する。
- `tf.Tensor` 互換のテンソル型（dtype / shape / データ I/O、tfjs に寄せた命名）。
- SavedModel / GraphDef のロードと推論実行。
- **TS 側で実装する**薄い高レベル学習 API（GradientTape 相当 / SGD・Adam / `minimize`）。
- prebuild によるゼロコンパイル install。

**含まない（C API に露出しないため、構造上スコープ外）**

- Keras の高レベル層構築 API 全体 / `tf.data` パイプライン / `tf.function`（autograph）/ distribution strategy。
- ブラウザ / WASM / WebGL バックエンド（tfjs 本体の領分）。
- Python ランタイムへの依存。

---

## 2. アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│ Public TS API  (src/index.ts) … re-exported with tfjs names   │
├───────────┬───────────┬───────────────┬──────────────────────┤
│ tensor/   │ ops/       │ training/     │ model/               │
│ Tensor    │ op wrappers│ GradientTape  │ SavedModel/GraphDef  │
│ dtype     │ (mostly    │ optimizers    │ load & run           │
│ shape I/O │  generated)│ minimize      │                      │
└───────────┴─────┬──────┴───────┬───────┴─────────┬────────────┘
                  │              │                 │
                  ▼  backend/native.ts is the single door that loads the addon
        ┌─────────────────────────────────────────────┐
        │ N-API addon (src/native/*.cc, node-addon-api)│
        │  tensor conversion / TFE_Execute / Status→exc│
        └───────────────────┬─────────────────────────┘
                            ▼  dynamic link
        libtensorflow C API (c/c_api.h, c/eager/c_api.h)
```

- **バインド方式は N-API C++ addon（node-addon-api + node-gyp）**。FFI は使わない。実行時解決は `node-gyp-build`、配布は `prebuildify`。
- addon は実行時に libtensorflow を dynamic link する。libtensorflow 本体は npm tarball に**同梱せず**、**install 時に取得**する（`scripts/install.mjs`、tfjs-node と同じダウンロード方式）。
- **op ラッパはコード生成が原則**。手書きは tensor / training / model のコア層に限定する。

---

## 3. ディレクトリ構成

```
src/
  index.ts               public entry (re-exports with tfjs-compatible names)
  backend/
    native.ts            the only door that loads the addon via node-gyp-build; exports typed NativeBinding
    dtype.ts             DType enum and TF <-> JS TypedArray mapping table
  native/                N-API addon (C++)
    binding.cc           Napi addon registration. version()/status smoke. TFE_Context creation
    tensor.cc/.h         JS TypedArray <-> TF_Tensor <-> TFE_TensorHandle conversion
    execute.cc/.h        TFE_Op construction / attr setting / TFE_Execute. Status -> exceptions
    ophandle.cc/.h       GC-tied release of handles (Napi::ObjectWrap or External + finalizer)
  tensor/
    tensor.ts            Tensor class (holds a handle, data()/dataSync()/array()/dispose())
    factory.ts           tensor()/scalar()/zeros()/ones() etc.
  ops/
    index.ts             bundles and re-exports generated, with hand-written overrides on top
    manual/              hand-written ops (hard-to-generate variadic/polymorphic cases)
    generated/           ★ generated output. Do NOT hand-edit (see RULES.md)
  training/
    tape.ts              GradientTape equivalent (TFE tape C API via the addon)
    optimizer.ts         Optimizer base + sgd/adam/rmsprop
  model/
    saved-model.ts       TF_LoadSessionFromSavedModel wrapper, signature resolution, run
    graph.ts             GraphDef loading
scripts/
  install.mjs            fetch libtensorflow (postinstall). Pinned URL/checksum, verified
  codegen/
    index.mjs            TF_GetAllOpList -> read OpList proto -> generate TS
    op-list.ts           schema types for the fetched OpList
    render.ts            template for "one op = one TS function"
binding.gyp              addon build definition
```

---

## 4. データ型マッピング

対応表は `src/backend/dtype.ts` に 1 か所だけ置き、addon と TS で共有します。

| TF DType | 名前 | TS 表現（read/write） | 備考 |
|---|---|---|---|
| `TF_FLOAT` | float32 | `Float32Array` | 既定 |
| `TF_DOUBLE` | float64 | `Float64Array` | |
| `TF_INT32` | int32 | `Int32Array` | |
| `TF_INT64` | int64 | `BigInt64Array` | JS number では欠損するため BigInt |
| `TF_UINT8` | uint8 | `Uint8Array` | 画像など |
| `TF_BOOL` | bool | `Uint8Array`（0/1） | |
| `TF_STRING` | string | `string[]` / `Uint8Array[]` | 可変長。専用パス |
| `TF_COMPLEX64` | complex64 | `Float32Array`（interleaved） | 実部・虚部を交互配置 |

- 既定 dtype は `float32`。`int64` は必ず BigInt 系で扱う（tfjs は int32 止まりなのでここで拡張）。
- shape は `number[]`（空配列＝スカラー）。

---

## 5. ネイティブ層の契約（NativeBinding I/F）

`backend/native.ts` は addon を読み込み、以下の**安定した最小インターフェース**として型付けします。上位層（tensor/ops/training/model）は addon の生関数を直接触らず、必ずこの型経由で呼びます。

```ts
export interface NativeBinding {
  version(): string;                              // TF_Version()
  // context
  createContext(opts?: ContextOptions): Ctx;      // TFE_NewContext
  deleteContext(ctx: Ctx): void;
  // tensor <-> handle
  createHandle(ctx: Ctx, data: ArrayBufferView, shape: number[], dtype: number): Handle;
  handleShape(h: Handle): number[];
  handleDType(h: Handle): number;
  handleData(h: Handle): ArrayBufferView;         // TFE_TensorHandleResolve -> copy out
  deleteHandle(h: Handle): void;
  // eager execute (the single door for running every op)
  execute(ctx: Ctx, opName: string, inputs: Handle[], attrs: AttrMap): Handle[];
  // gradient
  gradient(ctx: Ctx, ys: Handle[], xs: Handle[], dys?: Handle[]): Handle[];
  // op registry (for codegen)
  getAllOps(): Uint8Array;                         // serialized proto from TF_GetAllOpList
  // model
  loadSavedModel(ctx: Ctx, dir: string, tags: string[]): Model;
  runSignature(m: Model, sig: string, feeds: Record<string, Handle>): Record<string, Handle>;
}
```

- `Ctx` / `Handle` / `Model` は不透明ハンドル（`External`）。**必ず GC finalizer で解放**する。
- `execute` は全 op 実行のボトルネック。op 層のラッパは属性を組んでこれを呼ぶだけにする。
- 例外: Status が `TF_OK` 以外なら `TFError`（code + message）を throw する。無言で握り潰さない。

---

## 6. Op コード生成パイプライン

**1,400 個の op を手書きしません。** 実行時の op registry から生成します。

1. binding の `getAllOps()` が `TF_GetAllOpList()` の `OpList` proto（バイナリ）を返す。
2. `scripts/codegen/index.mjs` が proto をデコードし、各 `OpDef` について、入力（`input_arg`）を引数（`Tensor` / `Tensor[]`）に、属性（`attr`）を options 引数（`type`/`shape`/`int`/`bool`/`float`/`string`/リスト）に、出力（`output_arg`）を戻り値の tuple / 配列に解決し、`src/ops/generated/<snake>.ts` を 1 op = 1 関数で出力する。
3. 命名: op `MatMul` → 関数 `matMul`（tfjs にある名は tfjs のシグネチャに寄せ、無い op は snake→camel）。
4. 型: DType 属性は `DType` union、shape 属性は `number[]`、可変長入力は `Tensor[]`。
5. 生成物は `manual/` の手書き override より**下**に置き、衝突時は手書きを優先する。

**不変条件**: `generated/` は再生成で常に安全に上書きできること。手修正はテンプレート（`render.ts`）側に入れる。

---

## 7. メモリ管理

- `Tensor` は 1 個の `Handle` を保持する。`dispose()` が `deleteHandle` を呼ぶ。
- `tidy(fn)` を提供する: スコープ内で生成された中間 `Tensor` を関数終了時に一括破棄し、戻り値は残す。
- addon 側はすべての `TF_*`/`TFE_*` リソースに finalizer を張るので、JS 側で dispose を忘れても GC 時に解放される。ただし**早期 dispose が正**（GPU / 大テンソルの寿命を GC 任せにしない）。
- `execute` の入力ハンドルは所有権を移さない（呼び出し側が管理）。出力は新規所有。

---

## 8. 学習（training/）

公開 C API に **eager の勾配テープは無い**（`TFE_*Tape*` シンボルが存在しないことを実測で確認）。勾配 API は `TF_Graph` に対する `TF_AddGradients` のみ。そこで勾配は **記録＋リプレイ**（`tf.function` 相当）で実装します。

1. `tape.ts`: テープが有効な間、すべての op を記録する。記録は ops 層の `runOp` に相乗りさせるため、約 1,300 op すべてが op ごとの追加実装ゼロでトレース対象になる。テープが無いときは null チェック1つのみ。
2. `gradients.ts`: 記録を `TF_Graph` に再構築する。微分対象の入力（`xs`）は Placeholder に、それ以外の葉は eager の値から Const として焼き込む。`TF_AddGradients` でグラフを微分し、**TF 自身の勾配定義**（2.10 で 140 op 登録済み）をそのまま使う（手書きしない）。
3. セッションで forward と backward をまとめて実行し、結果は eager の `Tensor` で返る。公開 API は `grads(fn, xs)` と `valueAndGrads(fn, xs)`。

`TF_AddGradients` の結果は3種類あり、すべて明示的に扱う: 勾配ノード（正常）、`REGISTER_NO_GRADIENT_OP` の op（44 個、例 `Floor`）は **null**、到達不能な入力や勾配未登録 op は **Status エラー**。「使える勾配が無い」場合は必ず表面化させ、黙って 0 にはしない。

短命な補助テンソル（リダクションの軸、reshape の shape）を作る op は `disposeTemporary` に渡し、テープ記録中はリプレイ後まで生存させる。

Optimizer（`sgd` / `adam` / `rmsprop`、`minimize`）は **TS 側で eager op を使って実装**し、状態（moment・step）は `Tensor` として `Variable` ラッパ経由で更新する。`compileGrads`/`runGrads` でトレースと実行を分離し、`Optimizer.minimize` は損失グラフを (損失関数, 入力シグネチャ) 単位でキャッシュするため、学習ループは初回以降グラフを再構築しない。損失関数内で参照する変数以外のテンソルは初回トレース時に定数として焼き込まれる（フルバッチ前提）。shape が変わると再トレースする。

---

## 9. Model（model/）

- `loadSavedModel(dir, {tags=['serve']})` → `SavedModel`。`TF_LoadSessionFromSavedModel` を使う。
- signature（`serving_default` 等）を列挙し、`model.predict(feeds)` / `model.run(sig, feeds)` を提供する。
- feeds/fetches は名前ベース。Tensor ↔ handle 変換はネイティブ層に集約する。
- GraphDef 単体のロード（`graph.ts`）は SavedModel の下位機能として実装する。

---

## 10. libtensorflow の取得（scripts/install.mjs）

- libtensorflow を npm tarball に同梱しない（サイズ / ライセンス / アーキ差のため）。
- `postinstall` で、現在の OS/arch/（CPU|GPU）向けの**公式 libtensorflow アーカイブを取得・展開**する。
- **URL とアーカイブの checksum を固定し検証**する（改竄・取り違え防止）。バージョンは 1 か所で管理する。
- **バージョン固定（2.10.0）とヘッダ完全性。** 既定は libtensorflow 2.10.0。全プラットフォームで C API ヘッダが自己完結しているため。2.12+ の新しいアーカイブは Windows で分離ヘッダ（`tf_buffer.h` と `tsl/*` ツリー）が欠落しビルドが壊れる。安全策として、`install.mjs` は展開後に参照されているのに欠けている `tensorflow/*.h` を同一タグのソースから取得する（完全なアーカイブでは no-op）。2.11 より新しい既定に上げるには `tsl/*` のヘッダ欠落解決が必要で、将来課題。
- ダウンロード元が無い環境向けに、既存の libtensorflow を `LIBTENSORFLOW_ROOT` で指す抜け道を用意する。
- **Bun で install する利用者への注意**: Bun は依存の `postinstall`/`install` を既定でブロックする。利用側 `package.json` の `trustedDependencies` に `"tfjs-native"` を追加する必要がある（README に明記）。取得済み libtensorflow を指す `LIBTENSORFLOW_ROOT` でも回避できる。

---

## 11. ビルド・テスト・配布

### 開発ランタイム / パッケージマネージャ

- **開発は Bun**（PM＋ランタイム＋テストランナー）。ロックは `bun.lock`。
- **PM バージョンを固定しない**（`packageManager` フィールドを置かない）。どのバージョンでも同じ挙動を前提とする。
- ネイティブ addon は **N-API** なので、ビルドした prebuild は **Node でも Bun でも**ロードできる。

### コマンド（bun）

```bash
bun install          # resolve dependencies (bun.lock)
bun run build        # tsdown: TS -> dist (ESM .mjs + CJS .cjs + both d.ts)
bun run build:native # node-gyp rebuild (local addon build)
bun run prebuildify  # prebuildify: build a prebuild for the current OS/arch into prebuilds/
                     # ("prebuild" collides with the build pre-hook name, so it is unused)
bun run codegen      # scripts/codegen: regenerate all op wrappers
bun test             # tests (bun's built-in runner, imports from "bun:test")
bun run typecheck    # tsc --noEmit
bun run check        # biome check (lint + format verification)
bun run format       # biome format --write
```

### 出力フォーマット（ESM / CJS 両対応）

- tsdown が **ESM（`dist/index.mjs`）と CJS（`dist/index.cjs`）の両方**と、型 `index.d.mts` / `index.d.cts` を出力する。`package.json` の `exports` は `import`/`require` 条件で振り分ける（`main`=cjs, `module`=mjs, `types`=d.cts）。
- CJS でも動くよう、addon ローダは `import.meta.url` に依存しつつ **`binding.gyp` を上方探索**して package root を解決する（`src/backend/native.ts`）。dist（1 階層）と src（2 階層）の差、bundle 後の CJS `__filename` shim のどちらでも正しく root を指す。

### 技術スタック

- **TypeScript@6**（`strict`、ESM ソース）。※ npm 上の最新は 7 系だが、本プロジェクトは明示的に 6 系を固定する。
- **tsdown**（rolldown ベース）で **ESM+CJS の二重出力**、**Biome** で lint/format、**bun test** でテスト。
- **Node.js 22 以上が必須**（`engines.node >= 22`）。
- ネイティブ: **node-addon-api / node-gyp / node-gyp-build / prebuildify**。

### CI / Release（運用の詳細は RULES.md）

- **CI**: `oven-sh/setup-bun` で matrix（Linux/macOS/Windows）× bun。addon build（M5 以降）→ typecheck → biome → `bun test`。
- **Release**: tag `v*` で各 OS の prebuild を生成・集約し、**npm trusted publishing（OIDC、token 無し）**で publish。provenance は明示付与。publish 手順のみ npm CLI を使う（bun publish は OIDC 未対応のため）。

---

## 12. ロードマップ

1. **M0 スケルトン**: addon が libtensorflow をロードし `version()` を返す smoke test。完了。
2. **M1 Tensor**: 完了。dtype/shape、JS↔TF のデータ往復、`data()`/`dataSync()`/`array()`、および `tidy`。
3. **M2 Eager ops**: 完了。native `execute` + TFE_Context + 手書き op セット（M2a）に加え、protobufjs による `TF_GetAllOpList` から約 1,295 ラッパを生成（M2b）。生成ファイル（`src/ops/generated/index.ts`）はコミット済みで、libtensorflow 更新時に再生成する。
4. **M3 SavedModel**: 完了。addon 側で `TF_LoadSessionFromSavedModel` + `TF_SessionRun` を実装し、signature は MetaGraphDef を自前の小さな protobuf リーダで解析（実行時依存をゼロのまま維持）。
5. **M4 学習**: 完了。記録＋リプレイによる勾配（`grads`/`valueAndGrads`）、`Variable`、`sgd`/`adam`/`rmsprop`、損失グラフをキャッシュする `minimize`。§8 参照。
6. **M5 配布**: prebuild マトリクス + libtensorflow 自動取得は CI で 3 プラットフォームとも緑。残るは npm への trusted-publish リリースのみ。← 現在地

## 13. 用語

- **op registry**: TF がビルド時に登録する演算定義の集合（`TF_GetAllOpList` で取得可能）。
- **eager 実行**: グラフを組まず即時に op を評価する方式（`TFE_*`）。本 wrapper の基本実行モデル。
- **handle**: `TFE_TensorHandle*` などネイティブ資源への不透明参照。JS では `External` で表す。
