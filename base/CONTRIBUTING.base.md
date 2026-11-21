# Contributing to tfjs-native

:::kiritan{locale=en}
Thanks for your interest! This guide covers the developer setup and workflow. For the project design read [docs/DESIGN.md](./docs/DESIGN.md); for the mandatory conventions (commits, language, release) read [docs/RULES.md](./docs/RULES.md).

All repo artifacts and communication are in **English** (see the language policy in [docs/RULES.md](./docs/RULES.md)). The English and Japanese documents are generated from `base/` with Kiritan; edit the base files (see the localization rules in [docs/RULES.md](./docs/RULES.md)).
:::

:::kiritan{locale=ja}
貢献に興味を持っていただきありがとうございます。本ガイドは開発環境のセットアップとワークフローを扱います。設計は [docs/ja/DESIGN.md](./docs/ja/DESIGN.md)、必須の規約（コミット・言語・リリース）は [docs/ja/RULES.md](./docs/ja/RULES.md) を参照してください。

リポジトリに残る成果物とコミュニケーションはすべて **英語** です（言語ポリシーは [docs/ja/RULES.md](./docs/ja/RULES.md)）。英語版と日本語版のドキュメントは Kiritan で `base/` から生成するので、base ファイルを編集してください([docs/ja/RULES.md](./docs/ja/RULES.md) のローカライズの項)。
:::

:::kiritan{locale=en}
## Prerequisites

Working on the **pure-TS** parts only needs **Bun**. Building the **native addon** additionally needs **uv**, a C++ toolchain, and libtensorflow.

| Tool | Version | Why | Install |
|---|---|---|---|
| **Bun** | latest | package manager + runtime + test runner (do not pin its version) | <https://bun.sh> |
| **Node.js** | >= 22 | ABI target for the prebuilt addon; also runs `node-gyp` | <https://nodejs.org> |
| **uv** | latest, **on PATH** | resolves the Python `node-gyp` needs, and runs `clang-format` for `check:cpp`/`format:cpp` | <https://docs.astral.sh/uv/getting-started/installation/> |
| **C++ toolchain** | per-OS | compiles the N-API addon | see [below](#c-toolchain-per-os) |
| **libtensorflow** | 2.15.0 (default) | the native C library the addon links | fetched automatically |

uv is a hard prerequisite: `bun run build:native`/`bun run setup` shell out to bare `uv`, and `bun run check:cpp`/`format:cpp` shell out to bare `uvx`, assuming both resolve via PATH — the official installer adds them there by default. If you installed uv somewhere it doesn't put on PATH, add its `bin` directory to PATH yourself (Windows: `[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";<dir>", "User")`, then open a new shell).
:::

:::kiritan{locale=ja}
## 前提環境・前提ツール

**純粋 TS** の部分だけなら **Bun** だけで作業できます。**ネイティブ addon** をビルドする場合は追加で **uv**・C++ ツールチェーン・libtensorflow が必要です。

| ツール | バージョン | 用途 | 入手 |
|---|---|---|---|
| **Bun** | 最新 | パッケージマネージャ＋ランタイム＋テストランナー（バージョン固定しない） | <https://bun.sh> |
| **Node.js** | 22 以上 | prebuild addon の ABI ターゲット。`node-gyp` の実行にも使用 | <https://nodejs.org> |
| **uv** | 最新、**PATH に通す** | `node-gyp` が必要とする Python を解決し、`check:cpp`/`format:cpp` の `clang-format` 実行にも使用 | <https://docs.astral.sh/uv/getting-started/installation/> |
| **C++ ツールチェーン** | OS 別 | N-API addon のコンパイル | [下記](#os-別-c-ツールチェーン) |
| **libtensorflow** | 2.15.0(既定) | addon がリンクするネイティブ C ライブラリ | 自動取得 |

uv は必須の前提ツールです。`bun run build:native`/`bun run setup` は素の `uv` を、`bun run check:cpp`/`format:cpp` は素の `uvx` を、いずれも **PATH 解決前提**で呼び出します。公式インストーラなら既定で PATH に追加されます。カスタムな場所に入れて PATH に無い場合は、その `bin` ディレクトリを自分で PATH に追加してください（Windows: `[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";<dir>", "User")` の後、新しいシェルを開く）。
:::

:::kiritan{locale=en}
### Python via uv

```sh
# install uv (official installer), then a managed CPython:
uv python install    # reads .python-version (3.12)
uv python find 3.12  # prints the python.exe / python path
```

The interpreter version is pinned in `.python-version`. There is **no `uv.lock`**: the project has no Python dependencies — uv only provides a CPython for `node-gyp`.

`bun run build:native` (and `bun run setup`) resolve this Python automatically via `scripts/lib/python.mjs` — no manual `PYTHON` export needed. To do it yourself instead:
:::

:::kiritan{locale=ja}
### uv による Python

```sh
# uv（公式インストーラ）を入れてから、管理下の CPython を導入:
uv python install    # .python-version（3.12）を読む
uv python find 3.12  # python.exe / python のパスを表示
```

インタプリタのバージョンは `.python-version` に固定しています。**`uv.lock` はありません**。本プロジェクトに Python 依存は無く、uv は `node-gyp` 用の CPython を用意するだけだからです。

`bun run build:native`（および `bun run setup`）は `scripts/lib/python.mjs` 経由でこの Python を自動解決します。手動で `PYTHON` を export する必要はありません。自分で行いたい場合:
:::

```sh
# Windows (PowerShell)
$env:PYTHON = (uv python find 3.12)
# macOS / Linux
export PYTHON="$(uv python find 3.12)"
```

:::kiritan{locale=en}
### C++ toolchain per OS

- **Windows**: Visual Studio Build Tools 2022 with the *Desktop development with C++* workload (provides MSVC + the Windows SDK, which `node-gyp` drives via MSBuild).
- **Linux**: `build-essential` (gcc/g++, make).
- **macOS**: Xcode Command Line Tools (`xcode-select --install`).
:::

:::kiritan{locale=ja}
### OS 別 C++ ツールチェーン

- **Windows**: Visual Studio Build Tools 2022 の *Desktop development with C++* ワークロード（MSVC + Windows SDK を提供。`node-gyp` は MSBuild 経由で使用）。
- **Linux**: `build-essential`（gcc/g++, make）。
- **macOS**: Xcode Command Line Tools（`xcode-select --install`）。
:::

### libtensorflow

:::kiritan{locale=en}
`scripts/install.mjs` downloads the official C library for your platform to `deps/libtensorflow/`. To fetch it explicitly:

```sh
node scripts/install.mjs
```

Overrides:

- `LIBTENSORFLOW_ROOT` — use an existing install (must contain `include/` and `lib/`). Required on **macOS arm64** (`brew install libtensorflow`), which the official bucket does not provide.
- `TFJS_NATIVE_CDN_STORAGE` — mirror base URL.
- `TFJS_NATIVE_LIBTENSORFLOW_VERSION` — pin a different version.
- `TFJS_NATIVE_SKIP_INSTALL=1` — skip the fetch entirely.
:::

:::kiritan{locale=ja}
`scripts/install.mjs` がお使いのプラットフォーム向けの公式 C ライブラリを `deps/libtensorflow/` にダウンロードします。明示的に取得するには:

```sh
node scripts/install.mjs
```

上書き用の環境変数:

- `LIBTENSORFLOW_ROOT` — 既存インストールを使用（`include/` と `lib/` を含むこと）。公式バケットに無い **macOS arm64** では必須（`brew install libtensorflow`）。
- `TFJS_NATIVE_CDN_STORAGE` — ミラーのベース URL。
- `TFJS_NATIVE_LIBTENSORFLOW_VERSION` — 別バージョンを指定。
- `TFJS_NATIVE_SKIP_INSTALL=1` — 取得を完全にスキップ。
:::

:::kiritan{locale=en}
## Setup

One command does everything — installs JS dependencies, resolves the pinned Python through uv, fetches libtensorflow, builds the native addon, and verifies it loads:

```sh
bun run setup
```

It is safe to re-run. **Windows needs no `PATH` setup**: the addon loader prepends the fetched `deps/libtensorflow/lib` before loading, and the setup script verifies that end to end.

The equivalent manual steps, if you prefer to run them yourself:

```sh
TFJS_NATIVE_SKIP_INSTALL=1 bun install
node scripts/install.mjs
bun run build:native   # resolves Python via uv on its own; set PYTHON yourself to override
```
:::

:::kiritan{locale=ja}
## セットアップ

コマンド1つで完了します。JS 依存のインストール、uv による固定 Python の解決、libtensorflow の取得、ネイティブ addon のビルド、ロード確認までを行います:

```sh
bun run setup
```

再実行しても安全です。**Windows でも `PATH` の設定は不要**です。addon ローダが取得済みの `deps/libtensorflow/lib` を読み込み前に追加し、セットアップスクリプトがその動作まで検証します。

手動で実行したい場合の同等の手順:

```sh
TFJS_NATIVE_SKIP_INSTALL=1 bun install
node scripts/install.mjs
bun run build:native   # uv 経由で Python を自動解決。上書きしたい場合は自分で PYTHON を設定
```
:::

:::kiritan{locale=en}
## Everyday commands

```sh
bun run build       # tsdown: TS -> dist (ESM + CJS + d.ts)
bun run build:native# node-gyp rebuild (local addon)
bun run prebuildify # build a prebuilt binary for the current OS/arch
bun run codegen     # regenerate op wrappers from the TF op registry
bun test            # run tests (bun:test)
bun run typecheck   # tsc --noEmit
bun run lint        # biome lint only (no format check)
bun run check       # biome lint + format check (src/, scripts/, root configs)
bun run check:cpp   # clang-format check (src/native) — runs clang-format via uvx
bun run check:all   # check + check:cpp (everything)
bun run format      # biome format --write (src/, scripts/, root configs)
bun run format:cpp  # clang-format -i (src/native)
bun run format:all  # format + format:cpp
bun run docs:build  # regenerate README/CONTRIBUTING/docs from base/ (Kiritan)
bun run docs:check  # fail on a missing/stale translation or an output that drifted from base/
```

`check`/`lint` (biome) cover all JS/TS/JSON under `src/`, `scripts/`, and the root configs; `check:cpp` covers the C++ addon with a pinned `clang-format` (config in `.clang-format`, run via `uvx` — no separate install). `check:all` runs both. CI runs biome on every OS and the C++ check on Linux.

`bun run prebuildify` calls prebuildify's JS API directly (`scripts/prebuildify.mjs`) rather than its CLI: prebuildify's CLI shells out to the literal name `node-gyp.cmd` on Windows, but bun's `node_modules/.bin` shims there are `.exe`/`.bunx`, not `.cmd`, so the CLI form fails with `'node-gyp.cmd' is not recognized`. The script passes an explicit path instead, and resolves Python the same way `build:native` does.

Tests that need the compiled addon are **skip-guarded**: they run automatically once `bun run build:native` has produced the addon, and are skipped otherwise, so `bun test` stays green without a native build.
:::

:::kiritan{locale=ja}
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
bun run docs:build  # base/ から README・CONTRIBUTING・docs を再生成(Kiritan)
bun run docs:check  # 翻訳の欠落・古さや、base/ とずれた生成物があれば失敗
```

`check`・`lint`（biome）は `src/`・`scripts/`・ルート設定の JS/TS/JSON を、`check:cpp` は C++ addon を対象にします。C++ は固定バージョンの `clang-format`（設定は `.clang-format`、`uvx` 経由で実行するため別途インストール不要）を使います。`check:all` は両方を実行します。CI は biome を全 OS、C++ チェックを Linux で実行します。

`bun run prebuildify` は prebuildify の CLI ではなく **JS API を直接呼び出します**（`scripts/prebuildify.mjs`）。prebuildify の CLI は Windows で `node-gyp.cmd` という決め打ちの名前を呼び出しますが、bun が生成する `node_modules/.bin` のシムは `.exe`/`.bunx` であり `.cmd` ではないため、CLI 形式だと `'node-gyp.cmd' is not recognized` で失敗します。このスクリプトは明示的なパスを渡すことでこれを回避し、Python の解決も `build:native` と同じ方法で行います。

コンパイル済み addon を要するテストは **skip 保護**されています。`bun run build:native` で addon が生成されると自動的に実行され、無ければスキップされるので、ネイティブビルド無しでも `bun test` は緑のままです。
:::

:::kiritan{locale=en}
## Adding a platform / bumping libtensorflow

Checksums are pinned in `scripts/install.mjs`. To add one:

1. Set `TFJS_NATIVE_LIBTENSORFLOW_VERSION` (and platform, if cross-fetching) and run `node scripts/install.mjs`.
2. The installer prints `sha256(<artifact>) = <hash>` for unpinned artifacts.
3. Add that entry to the `CHECKSUMS` map and commit.

Bumping the default version follows the **libtensorflow Version Policy** in [docs/RULES.md](./docs/RULES.md) (how it maps to a tfjs-native version bump).
:::

:::kiritan{locale=ja}
## プラットフォーム追加 / libtensorflow のバージョン更新

チェックサムは `scripts/install.mjs` にピン止めしています。追加手順:

1. `TFJS_NATIVE_LIBTENSORFLOW_VERSION`（必要ならプラットフォームも）を設定して `node scripts/install.mjs` を実行。
2. 未ピンのアーティファクトについてインストーラが `sha256(<artifact>) = <hash>` を表示。
3. その値を `CHECKSUMS` マップに追加してコミット。

既定バージョンの引き上げは [docs/ja/RULES.md](./docs/ja/RULES.md) の **libtensorflow バージョンポリシー**（tfjs-native のバージョン更新への対応付け）に従います。
:::

:::kiritan{locale=en}
## Regenerating op wrappers

Op wrappers under `src/ops/generated/` are produced by `scripts/codegen/` from the runtime op registry (`TF_GetAllOpList`). **Do not hand-edit generated files** — change the generator/template and run `bun run codegen`.
:::

:::kiritan{locale=ja}
## op ラッパの再生成

`src/ops/generated/` の op ラッパは `scripts/codegen/` が実行時 op registry（`TF_GetAllOpList`）から生成します。**生成ファイルを手編集しない**でください。ジェネレータ/テンプレートを直して `bun run codegen` を実行します。
:::

:::kiritan{locale=en}
## Regenerating test fixtures

The SavedModel tests use a tiny fixture in `test/fixtures/times_two` (y = x * 2). TensorFlow's Python package is needed only to *produce* it — tfjs-native never depends on it. Use Python 3.12 (TensorFlow has no 3.13/3.14 wheels):

```sh
uv venv --python 3.12 .venv-tf
uv pip install --python .venv-tf tensorflow-cpu
.venv-tf/Scripts/python scripts/fixtures/make_saved_model.py   # Windows
.venv-tf/bin/python scripts/fixtures/make_saved_model.py       # macOS / Linux
```

Those tests skip automatically when the fixture or the native addon is missing.
:::

:::kiritan{locale=ja}
## テスト fixture の再生成

SavedModel のテストは `test/fixtures/times_two`（y = x * 2）という極小 fixture を使います。TensorFlow の Python パッケージは fixture を**生成するときだけ**必要で、tfjs-native 自体は依存しません。Python 3.12 を使ってください（TensorFlow に 3.13/3.14 の wheel が無いため）:

```sh
uv venv --python 3.12 .venv-tf
uv pip install --python .venv-tf tensorflow-cpu
.venv-tf/Scripts/python scripts/fixtures/make_saved_model.py   # Windows
.venv-tf/bin/python scripts/fixtures/make_saved_model.py       # macOS / Linux
```

fixture かネイティブ addon が無い場合、これらのテストは自動的にスキップされます。
:::

:::kiritan{locale=en}
## Pull requests

- Branch from `main` (`feat/…`, `fix/…`, `chore/…`); never push to `main` directly.
- Commit messages follow **Conventional Commits**, in English (see [docs/RULES.md](./docs/RULES.md)). Example: `feat(ops): add scatterNd wrapper`.
- CI must be green: native build (where applicable), `typecheck`, `biome`, `bun test`.
- PRs that change the public API must include the reviewed `d.ts` impact.
- Keep generated code and hand-written changes in separate commits.
:::

:::kiritan{locale=ja}
## プルリクエスト

- `main` からブランチを切る（`feat/…`, `fix/…`, `chore/…`）。`main` へ直接 push しない。
- コミットメッセージは英語で **Conventional Commits** に従う（[docs/ja/RULES.md](./docs/ja/RULES.md)）。例: `feat(ops): add scatterNd wrapper`。
- CI が緑であること: native build（該当時）, `typecheck`, `biome`, `bun test`。
- 公開 API を変える PR は `d.ts` への影響をレビューに含める。
- 生成コードと手書き変更は別コミットに分ける。
:::

:::kiritan{locale=en}
## Releasing (maintainers)

Releases run `.github/workflows/release.yml`, triggered manually (`workflow_dispatch`) with a version input (a semver bump like `patch`, or an explicit `0.0.1`). It builds a prebuild on each OS, aggregates them, bumps the version, publishes to npm via **trusted publishing (OIDC)** — no `NPM_TOKEN` — then commits, tags `v<semver>`, and creates the GitHub Release. The npm package must be registered as a trusted publisher for this repo and workflow on npmjs.com before the first release.
:::

:::kiritan{locale=ja}
## リリース（メンテナ向け）

リリースは `.github/workflows/release.yml` を手動（`workflow_dispatch`）で起動します。version 入力（`patch` などの semver bump か、`0.0.1` のような明示指定）を渡すと、各 OS で prebuild をビルド・集約し、バージョンを上げて **trusted publishing（OIDC）** で npm に公開（`NPM_TOKEN` 不使用）、その後コミット・`v<semver>` タグ付け・GitHub Release 作成を行います。初回リリース前に、この npm パッケージを本リポジトリ＋ワークフローの trusted publisher として npmjs.com に登録しておく必要があります。
:::
