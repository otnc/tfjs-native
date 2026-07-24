# Operational Rules — tfjs-native

**English** | [日本語](./ja/RULES.md)

This file holds the **operational rules** (conventions both humans and Claude follow). For design, see [DESIGN.md](./DESIGN.md). The rules here are mandatory; deviations must be explicitly agreed on in a PR.

## Language Policy

**Principle: everything that lives in the repo — artifacts and communication — is in English.** The only Japanese kept in this repo is under `.private/` (untracked) and the explicit localizations (`*-ja.md`, `docs/ja/*`).

| Target | Language |
|---|---|
| Identifiers (variables, functions, type names) | **English only** |
| In-code comments / TSDoc | **English only** |
| Commit messages | **English only** (Conventional Commits, below) |
| PR title / body | **English only** |
| Issue title / body | **English only** |
| Code review comments | **English only** |
| README / docs / CONTRIBUTING etc. | **English is canonical** |
| `.private/` translations | Japanese allowed (untracked) |

### Localization (adding a Japanese version)

The canonical English file keeps its default name. To add a Japanese version, make it a **separate file** and signal it in the name.

| English (canonical) | Japanese version |
|---|---|
| `README.md` | `README-ja.md` |
| `CONTRIBUTING.md` | `CONTRIBUTING-ja.md` |
| `docs/foo.md` | `docs/ja/foo.md` |

- Do not mix Japanese into the default-named files (`README.md` etc.).
- A Japanese version links to the English one at the top; if they drift, update to match the English as the source of truth.

### Markdown line breaks

Do not hard-wrap prose — keep each paragraph and list item on a single source line. A hard wrap renders as a stray space (visibly wrong in Japanese text). Wrapping is fine only inside code blocks and tables.

## Commit Message Convention (Conventional Commits, English required)

Format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

- **type**: `feat` / `fix` / `docs` / `style` / `refactor` / `perf` / `test` / `build` / `ci` / `chore` / `revert`
- **scope** (optional): `tensor` / `ops` / `native` / `training` / `model` / `codegen` / `ci` / `release` etc.
- **subject**: English, imperative ("add", not "added"/"adds"), lowercase start, no trailing period, aim for <= 50 chars
- **body** (optional): why the change was made. Wrap at 72 chars
- **footer** (optional): `BREAKING CHANGE: ...` / `Closes #123`

Examples:
```
feat(ops): generate typed wrappers from TF_GetAllOpList
fix(native): free TFE_TensorHandle on dtype mismatch
```

- Breaking changes must carry `type(scope)!: ...` or a `BREAKING CHANGE:` footer.

### Commit granularity

**Commit at a fine granularity — one concern per commit.** A commit should be reviewable on its own and describable in a single subject line without "and".

- Split by concern, not by file: a dependency bump and a bug fix are two commits even when they touch the same file.
- Keep generated output (`src/ops/generated/`) in its own commit, separate from hand-written changes.
- Keep docs, CI, and source changes apart unless one is meaningless without the other.
- Never bundle unrelated work into a "wip" or "misc" commit.
- Push in the same small increments so CI reports on each step.

## Branches / PRs

- Never push directly to `main`. Branch names: `feat/…` `fix/…` `chore/…`.
- A PR requires **all green CI** (native build / typecheck / biome / `bun test`) to merge.
- Squash merge. The squashed title must also follow Conventional Commits.
- PRs that only contain generated code get the `codegen` label; review may go by the diff summary.

## Versioning / Release

- **SemVer** over tfjs-native's own public TS API.
- Releases fire on a pushed tag `v<semver>` (e.g. `v0.2.0`) via `release.yml`.
- **npm trusted publishing (OIDC) only.** Do not store or use an `NPM_TOKEN` in the repo.
- CHANGELOG is generated from Conventional Commits (manual additions allowed).
- Assemble prebuilds for all OSes before publishing (never publish with some missing).

## libtensorflow Version Policy

- The bundled libtensorflow version is **pinned** in `scripts/install.mjs` (`VERSION`), currently **2.10.0**. It is chosen for cross-platform header completeness (see [DESIGN.md](./DESIGN.md) §10), not for being the newest release.
- The libtensorflow version is **independent** of the tfjs-native version; SemVer is judged on tfjs-native's own TS API, not on which libtensorflow it links.
- **Bumping the pinned libtensorflow** maps to a tfjs-native version bump by impact: additive only (new ops, no TS API change) → **minor**; removes/renames ops or changes behavior users can depend on → **major**; a pure rebuild on the same libtensorflow → **patch**.
- Every libtensorflow bump must update `VERSION`, add the new archive checksums (`CHECKSUMS`), and state the libtensorflow version in the release notes.
- Consumers may override the version with `TFJS_NATIVE_LIBTENSORFLOW_VERSION`, but unpinned versions are best-effort (checksum unverified, headers unverified).

## Development Environment Rules

- Develop with **bun** (PM + runtime + test runner). Tests use `bun test` (`import ... from "bun:test"`).
- **Node.js 22+ required** (`engines.node >= 22`). Do not add compatibility code for older versions.
- **Do not pin the PM version via the `packageManager` field.** Keep the assumption that any version works.
- Artifacts are **ESM/CJS dual**. When adding public API, both outputs and both d.ts must stay intact (verify with `bun run build`).

## Code Quality Rules

- Formatting/linting is **Biome** (JS/TS/JSON) and **clang-format** (C++). Do not introduce ESLint / Prettier.
- Types are `strict`. `any` is forbidden in principle (at the native boundary where unavoidable, use `// biome-ignore` with a stated reason).
- **Do not hand-edit `src/**/generated/`.** Fix `scripts/codegen/` and regenerate.
- Native resources (`TF_*` / `TFE_*` handles) must be released within the same responsibility boundary that created them. Leaking PRs are rejected.
- PRs that change the public API must have their d.ts diff reviewed.

## Security / Secrets

- Pin and verify the libtensorflow download source and checksum (`scripts/install.mjs`).
- Keep no long-lived secrets in CI (publish uses OIDC; otherwise the least-privilege `GITHUB_TOKEN`).
