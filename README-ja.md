# tfjs-native

[English](./README.md) | **日本語**

[![CI](https://github.com/otnc/tfjs-native/actions/workflows/ci.yml/badge.svg)](https://github.com/otnc/tfjs-native/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/tfjs-native.svg)](https://www.npmjs.com/package/tfjs-native)
![node](https://img.shields.io/badge/node-%3E%3D22-brightgreen)
![platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

Node.js 向けの TensorFlow（C++）ネイティブ wrapper。

`@tensorflow/tfjs` がマップする TF op は約 300 個ですが、オリジナルの op registry には 1,400 以上あります。**tfjs-native** は TensorFlow の **C API**（`@tensorflow/tfjs-node` も使うネイティブライブラリ）をバインドし、**全 op registry** を型付き TypeScript から使えるようにします。加えて SavedModel の実行と薄い学習レイヤも目標です。

Original: [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)

> **ステータス: 開発初期。** テンソル層とネイティブバインディングは実装済み（マイルストーン M1）で、API と op カバレッジは拡張中です。npm 未公開。

## 動作要件

- **Node.js 22 以上**（**Windows / Linux / macOS**）。
- prebuild を配布するのは **linux-x64**・**darwin-x64**・**win32-x64** です。それ以外の環境ではソースからビルドします（Python と C++ ツールチェーンが必要）。
- `libtensorflow`（C ライブラリ）はインストール時に自動取得されます。対応環境では手動セットアップは不要です。

## インストール

```sh
npm install tfjs-native
# または: bun add tfjs-native
```

インストール時に、環境に合った `libtensorflow` がダウンロードされます。

- **インストールスクリプトの許可。** パッケージマネージャによっては依存のインストールスクリプトを既定で実行せず、ダウンロードがスキップされます。tfjs-native を許可するか、`LIBTENSORFLOW_ROOT` で既存のインストールを指定してください:
  - **Bun**: `trustedDependencies` に `"tfjs-native"` を追加。
  - **pnpm**: `pnpm.onlyBuiltDependencies` に `"tfjs-native"` を追加。
  - **npm**: 現状は実行しますが、今後 opt-in が必要になる見込みです。
- **macOS（Apple Silicon）**: 公式 arm64 ビルドが無いため、`brew install libtensorflow` の後に `LIBTENSORFLOW_ROOT` を設定してください。
- ミラーは `TFJS_NATIVE_CDN_STORAGE=https://your-mirror/` で指定します。

## 使い方

```ts
import { tensor, zeros, tidy, version } from "tfjs-native";

console.log(version()); // リンクされた libtensorflow のバージョン

const a = tensor([
  [1, 2, 3],
  [4, 5, 6],
]);
console.log(a.shape); // [2, 3]
console.log(await a.array()); // [[1, 2, 3], [4, 5, 6]]

// int64 を第一級サポート（BigInt で読み出し）。int32 止まりの tfjs との違い。
const ids = tensor([1, 2, 3], [3], "int64");

// スコープ単位のメモリ管理。
const result = tidy(() => {
  const z = zeros([2, 2]); // スコープ終了時に破棄
  return tensor([1, 1]); // 保持
});

a.dispose();
```

## 仕組み

```
TypeScript API  ->  N-API addon (node-addon-api)  ->  libtensorflow C API
```

addon は固定バージョンの libtensorflow（現在 2.10.0）をリンクし、プラットフォームごとにコンパイルされ、`node-gyp-build` 経由で prebuild バイナリとして配布されます。

## ドキュメント

- [機能一覧](./docs/ja/FUNCTIONS.md) — 現在の API とロードマップ。
- [設計](./docs/ja/DESIGN.md) — アーキテクチャ・スコープ・ネイティブ契約。
- [コントリビュート](./CONTRIBUTING-ja.md) — 開発環境（Bun、uv による Python、C++ ツールチェーン）とワークフロー。

## Author

otoneko. https://github.com/otnc

## ライセンス

[Apache 2.0](./LICENSE) ライセンスで提供されています。
