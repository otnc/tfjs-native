# Contributing to tfjs-native

[English](./CONTRIBUTING.md) | **日本語**

貢献に興味を持っていただきありがとうございます。本ガイドは開発環境のセットアップとワークフローを扱います。設計は [docs/ja/DESIGN.md](./docs/ja/DESIGN.md)、必須の規約（コミット・言語・リリース）は [docs/ja/RULES.md](./docs/ja/RULES.md) を参照してください。

リポジトリに残る成果物とコミュニケーションはすべて **英語** です（言語ポリシーは [docs/ja/RULES.md](./docs/ja/RULES.md)）。日本語ローカライズは別ファイルにします（`README-ja.md`、`docs/ja/*`）。

## 前提環境・前提ツール

**純粋 TS** の部分だけなら **Bun** だけで作業できます。**ネイティブ addon** をビルドする場合は追加で Python・C++ ツールチェーン・libtensorflow が必要です。

| ツール | バージョン | 用途 | 入手 |
|---|---|---|---|
| **Bun** | 最新 | パッケージマネージャ＋ランタイム＋テストランナー（バージョン固定しない） | <https://bun.sh> |
| **Node.js** | 22 以上 | prebuild addon の ABI ターゲット。`node-gyp` の実行にも使用 | <https://nodejs.org> |
| **Python** | 3.x（3.12 で確認） | `node-gyp` がビルド構成に必要 | [uv](https://docs.astral.sh/uv/getting-started/installation/) 経由（下記） |
| **C++ ツールチェーン** | OS 別 | N-API addon のコンパイル | [下記](#os-別-c-ツールチェーン) |
| **libtensorflow** | 2.10.0（既定） | addon がリンクするネイティブ C ライブラリ | 自動取得 |

### uv による Python（推奨）

```sh
# uv（公式インストーラ）を入れてから、管理下の CPython を導入:
uv python install    # .python-version（3.12）を読む
uv python find 3.12  # python.exe / python のパスを表示
```

インタプリタのバージョンは `.python-version` に固定しています。**`uv.lock` はありません**。本プロジェクトに Python 依存は無く、uv は `node-gyp` 用の CPython を用意するだけだからです。

addon をビルドするとき `node-gyp` にそのパスを渡します:

```sh
# Windows (PowerShell)
$env:PYTHON = (uv python find 3.12)
# macOS / Linux
export PYTHON="$(uv python find 3.12)"
```

### OS 別 C++ ツールチェーン

- **Windows**: Visual Studio Build Tools 2022 の *Desktop development with C++* ワークロード（MSVC + Windows SDK を提供。`node-gyp` は MSBuild 経由で使用）。
- **Linux**: `build-essential`（gcc/g++, make）。
- **macOS**: Xcode Command Line Tools（`xcode-select --install`）。

### libtensorflow

`scripts/install.mjs` がお使いのプラットフォーム向けの公式 C ライブラリを `deps/libtensorflow/` にダウンロードします。明示的に取得するには:

```sh
node scripts/install.mjs
```

上書き用の環境変数:

- `LIBTENSORFLOW_ROOT` — 既存インストールを使用（`include/` と `lib/` を含むこと）。公式バケットに無い **macOS arm64** では必須（`brew install libtensorflow`）。
- `TFJS_NATIVE_CDN_STORAGE` — ミラーのベース URL。
- `TFJS_NATIVE_LIBTENSORFLOW_VERSION` — 別バージョンを指定。
- `TFJS_NATIVE_SKIP_INSTALL=1` — 取得を完全にスキップ。

## セットアップ

```sh
# libtensorflow の取得を発火させずに JS 依存を入れる
TFJS_NATIVE_SKIP_INSTALL=1 bun install

# プラットフォーム向けの libtensorflow を取得
node scripts/install.mjs

# ネイティブ addon をビルド（Python と C++ ツールチェーンが必要）
PYTHON="$(uv python find 3.12)" bun run build:native
```

**Windows** では、addon ロード時に `tensorflow.dll` が解決できる必要があります。`deps/libtensorflow/lib` を `PATH` に追加するか、ビルドした `.node` の隣に `tensorflow.dll` をコピーしてください。

## 日常コマンド

```sh
bun run build       # tsdown: TS -> dist（ESM + CJS + d.ts）
bun run build:native# node-gyp rebuild（ローカル addon）
bun run prebuildify # 現 OS/arch の prebuild を生成
bun run codegen     # TF op registry から op ラッパを再生成
bun test            # テスト実行（bun:test）
bun run typecheck   # tsc --noEmit
bun run lint        # biome の lint のみ（format 検証なし）
bun run check       # biome の lint + format 検証（src/, scripts/, ルート設定）
bun run check:cpp   # clang-format 検証（src/native）— uvx 経由で実行
bun run check:all   # check + check:cpp（全部）
bun run format      # biome format --write（src/, scripts/, ルート設定）
bun run format:cpp  # clang-format -i（src/native）
bun run format:all  # format + format:cpp
```

`check`・`lint`（biome）は `src/`・`scripts/`・ルート設定の JS/TS/JSON を、`check:cpp` は C++ addon を対象にします。C++ は固定バージョンの `clang-format`（設定は `.clang-format`、`uvx` 経由で実行するため別途インストール不要）を使います。`check:all` は両方を実行します。CI は biome を全 OS、C++ チェックを Linux で実行します。

コンパイル済み addon を要するテストは **skip 保護**されています。`bun run build:native` で addon が生成されると自動的に実行され、無ければスキップされるので、ネイティブビルド無しでも `bun test` は緑のままです。

## プラットフォーム追加 / libtensorflow のバージョン更新

チェックサムは `scripts/install.mjs` にピン止めしています。追加手順:

1. `TFJS_NATIVE_LIBTENSORFLOW_VERSION`（必要ならプラットフォームも）を設定して `node scripts/install.mjs` を実行。
2. 未ピンのアーティファクトについてインストーラが `sha256(<artifact>) = <hash>` を表示。
3. その値を `CHECKSUMS` マップに追加してコミット。

既定バージョンの引き上げは [docs/ja/RULES.md](./docs/ja/RULES.md) の **libtensorflow バージョンポリシー**（tfjs-native のバージョン更新への対応付け）に従います。

## op ラッパの再生成

`src/ops/generated/` の op ラッパは `scripts/codegen/` が実行時 op registry（`TF_GetAllOpList`）から生成します。**生成ファイルを手編集しない**でください。ジェネレータ/テンプレートを直して `bun run codegen` を実行します。

## テスト fixture の再生成

SavedModel のテストは `test/fixtures/times_two`（y = x * 2）という極小 fixture を使います。TensorFlow の Python パッケージは fixture を**生成するときだけ**必要で、tfjs-native 自体は依存しません。Python 3.12 を使ってください（TensorFlow に 3.13/3.14 の wheel が無いため）:

```sh
uv venv --python 3.12 .venv-tf
uv pip install --python .venv-tf tensorflow-cpu
.venv-tf/Scripts/python scripts/fixtures/make_saved_model.py   # Windows
.venv-tf/bin/python scripts/fixtures/make_saved_model.py       # macOS / Linux
```

fixture かネイティブ addon が無い場合、これらのテストは自動的にスキップされます。

## プルリクエスト

- `main` からブランチを切る（`feat/…`, `fix/…`, `chore/…`）。`main` へ直接 push しない。
- コミットメッセージは英語で **Conventional Commits** に従う（[docs/ja/RULES.md](./docs/ja/RULES.md)）。例: `feat(ops): add scatterNd wrapper`。
- CI が緑であること: native build（該当時）, `typecheck`, `biome`, `bun test`。
- 公開 API を変える PR は `d.ts` への影響をレビューに含める。
- 生成コードと手書き変更は別コミットに分ける。

## リリース（メンテナ向け）

リリースは tag `v<semver>` の push で `.github/workflows/release.yml` が実行します。各 OS の prebuild をビルド・集約し、**trusted publishing（OIDC）** で npm に公開します（`NPM_TOKEN` 不使用）。タグ付け前に全プラットフォームの prebuild が揃っていることを確認してください。
