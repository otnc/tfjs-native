# 運用ルール — tfjs-native

[English](../RULES.md) | **日本語**

このファイルは**運用ルール**（人間・Claude 双方が従う規約）をまとめたものです。設計は [DESIGN.md](./DESIGN.md) を参照してください。ここに書かれたルールは必須で、逸脱する場合は PR で明示的に合意してください。

## 言語ポリシー

**原則: リポジトリに残る成果物・コミュニケーションはすべて英語。** リポジトリ内で日本語を残すのは `.private/`（untracked）と、明示的にローカライズしたファイル（`*-ja.md`、`docs/ja/*`）だけです。

| 対象 | 言語 |
|---|---|
| 識別子（変数・関数・型名） | **英語のみ** |
| コード内コメント / TSDoc | **英語のみ** |
| コミットメッセージ | **英語のみ**（下記 Conventional Commits） |
| PR タイトル / 本文 | **英語のみ** |
| Issue タイトル / 本文 | **英語のみ** |
| コードレビューのコメント | **英語のみ** |
| README / docs / CONTRIBUTING 等 | **英語を正**とする |
| `.private/` の翻訳 | 日本語可（untracked） |

### ローカライズ（日本語版の追加）

正となる英語ファイルは既定名のまま置きます。日本語版を足すときは**別ファイル**にし、名前で示します。

| 英語（正） | 日本語版 |
|---|---|
| `README.md` | `README-ja.md` |
| `CONTRIBUTING.md` | `CONTRIBUTING-ja.md` |
| `docs/foo.md` | `docs/ja/foo.md` |

- 既定名のファイル（`README.md` 等）に日本語を混在させない。
- 日本語版は冒頭で英語版へリンクする。内容が乖離した場合は英語版を正として合わせる。

### Markdown の改行

散文をハードラップしない（段落・箇条書きは1行に収める）。ハードラップは余計な半角スペースとして描画され、日本語では特に不自然になる。改行してよいのはコードブロックとテーブルの中だけ。

## コミットメッセージ規約（Conventional Commits、英語必須）

書式:

```
<type>(<scope>): <subject>

<body>

<footer>
```

- **type**: `feat` / `fix` / `docs` / `style` / `refactor` / `perf` / `test` / `build` / `ci` / `chore` / `revert`
- **scope**（任意）: `tensor` / `ops` / `native` / `training` / `model` / `codegen` / `ci` / `release` など
- **subject**: 英語・命令法（"added"/"adds" ではなく "add"）・小文字始まり・末尾ピリオドなし・50 字以内を目安
- **body**（任意）: なぜ変えたか。72 字で折り返す
- **footer**（任意）: `BREAKING CHANGE: ...` / `Closes #123`

例:
```
feat(ops): generate typed wrappers from TF_GetAllOpList
fix(native): free TFE_TensorHandle on dtype mismatch
```

- 破壊的変更は `type(scope)!: ...` か `BREAKING CHANGE:` footer を必ず付ける。
- 1 コミット 1 論点。生成物（`src/ops/generated/`）と手書きの変更は別コミットに分ける。

## ブランチ / PR

- `main` へ直接 push しない。ブランチ名は `feat/…` `fix/…` `chore/…`。
- マージには **CI 全緑**（native build / typecheck / biome / `bun test`）が必須。
- Squash merge。squash 後のタイトルも Conventional Commits に従う。
- 生成コードのみの PR には `codegen` ラベルを付ける。レビューは差分サマリで可。

## バージョニング / リリース

- tfjs-native 自身の公開 TS API に対して **SemVer**。
- リリースは tag `v<semver>`（例 `v0.2.0`）の push で `release.yml` が実行。
- **npm trusted publishing（OIDC）のみ**。`NPM_TOKEN` をリポジトリに置かない・使わない。
- CHANGELOG は Conventional Commits から生成（手書き追記可）。
- publish 前に全 OS の prebuild を揃える（欠けたまま publish しない）。

## libtensorflow バージョンポリシー

- 同梱する libtensorflow のバージョンは `scripts/install.mjs`（`VERSION`）に**固定**しています。現在は **2.10.0**。最新版ではなく、全プラットフォームでのヘッダ完全性で選んでいます（[DESIGN.md](./DESIGN.md) §10 参照）。
- libtensorflow のバージョンと tfjs-native のバージョンは**独立**です。SemVer は tfjs-native 自身の TS API で判断し、どの libtensorflow をリンクするかでは判断しません。
- **固定 libtensorflow の引き上げ**は影響で tfjs-native のバージョンに対応付けます: 追加のみ（op 追加・TS API 変更なし）→ **minor**; op の削除・改名や、利用者が依存し得る挙動変更 → **major**; 同一 libtensorflow での純粋な再ビルド → **patch**。
- 引き上げ時は必ず `VERSION` を更新し、新しいアーカイブの checksum（`CHECKSUMS`）を追加し、リリースノートに libtensorflow のバージョンを明記する。
- 利用者は `TFJS_NATIVE_LIBTENSORFLOW_VERSION` でバージョンを上書きできますが、未固定のバージョンはベストエフォート（checksum 未検証・ヘッダ未検証）です。

## 開発環境ルール

- 開発は **bun**（PM＋ランタイム＋テストランナー）。テストは `bun test`（`import ... from "bun:test"`）。
- **Node.js 22 以上が必須**（`engines.node >= 22`）。それ未満向けの互換コードは足さない。
- **`packageManager` フィールドで PM バージョンを固定しない**。どのバージョンでも動く前提を崩さない。
- 配布物は **ESM/CJS 両対応**。公開 API を足すときは両出力・両 d.ts が壊れないこと（`bun run build` で確認）。

## コード品質ルール

- フォーマット / リントは **Biome**（JS/TS/JSON）と **clang-format**（C++）。ESLint / Prettier は導入しない。
- 型は `strict`。`any` は原則禁止（ネイティブ境界で已むを得ない箇所は `// biome-ignore` と理由を必ず添える）。
- **`src/**/generated/` を手編集しない**。修正は `scripts/codegen/` を直して再生成する。
- ネイティブリソース（`TF_*` / `TFE_*` ハンドル）は生成した責務境界と同じ場所で必ず解放する。リークする PR は却下。
- 公開 API を変える PR は d.ts の差分をレビューする。

## セキュリティ / 秘匿

- libtensorflow のダウンロード元・チェックサムは固定し検証する（`scripts/install.mjs`）。
- CI に長期シークレットを置かない（publish は OIDC、他は最小権限の `GITHUB_TOKEN`）。
