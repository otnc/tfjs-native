# Contributing to tfjs-native

**English** | [日本語](./CONTRIBUTING-ja.md)

Thanks for your interest! This guide covers the developer setup and workflow. For the project design read [docs/DESIGN.md](./docs/DESIGN.md); for the mandatory conventions (commits, language, release) read [docs/RULES.md](./docs/RULES.md).

All repo artifacts and communication are in **English** (see the language policy in [docs/RULES.md](./docs/RULES.md)). Japanese localizations are separate files (`README-ja.md`, `docs/ja/*`).

## Prerequisites

Working on the **pure-TS** parts only needs **Bun**. Building the **native addon** additionally needs Python, a C++ toolchain, and libtensorflow.

| Tool | Version | Why | Install |
|---|---|---|---|
| **Bun** | latest | package manager + runtime + test runner (do not pin its version) | <https://bun.sh> |
| **Node.js** | >= 22 | ABI target for the prebuilt addon; also runs `node-gyp` | <https://nodejs.org> |
| **Python** | 3.x (3.12 tested) | required by `node-gyp` to configure the build | via [uv](https://docs.astral.sh/uv/getting-started/installation/) (below) |
| **C++ toolchain** | per-OS | compiles the N-API addon | see [below](#c-toolchain-per-os) |
| **libtensorflow** | 2.10.0 (default) | the native C library the addon links | fetched automatically |

### Python via uv (recommended)

```sh
# install uv (official installer), then a managed CPython:
uv python install    # reads .python-version (3.12)
uv python find 3.12  # prints the python.exe / python path
```

The interpreter version is pinned in `.python-version`. There is **no `uv.lock`**: the project has no Python dependencies — uv only provides a CPython for `node-gyp`.

Point `node-gyp` at it when building the addon:

```sh
# Windows (PowerShell)
$env:PYTHON = (uv python find 3.12)
# macOS / Linux
export PYTHON="$(uv python find 3.12)"
```

### C++ toolchain per OS

- **Windows**: Visual Studio Build Tools 2022 with the *Desktop development with C++* workload (provides MSVC + the Windows SDK, which `node-gyp` drives via MSBuild).
- **Linux**: `build-essential` (gcc/g++, make).
- **macOS**: Xcode Command Line Tools (`xcode-select --install`).

### libtensorflow

`scripts/install.mjs` downloads the official C library for your platform to `deps/libtensorflow/`. To fetch it explicitly:

```sh
node scripts/install.mjs
```

Overrides:

- `LIBTENSORFLOW_ROOT` — use an existing install (must contain `include/` and `lib/`). Required on **macOS arm64** (`brew install libtensorflow`), which the official bucket does not provide.
- `TFJS_NATIVE_CDN_STORAGE` — mirror base URL.
- `TFJS_NATIVE_LIBTENSORFLOW_VERSION` — pin a different version.
- `TFJS_NATIVE_SKIP_INSTALL=1` — skip the fetch entirely.

## Setup

```sh
# install JS deps without triggering the libtensorflow fetch yet
TFJS_NATIVE_SKIP_INSTALL=1 bun install

# fetch libtensorflow for your platform
node scripts/install.mjs

# build the native addon (needs Python + the C++ toolchain)
PYTHON="$(uv python find 3.12)" bun run build:native
```

On **Windows**, the runtime needs `tensorflow.dll` to be discoverable when the addon loads — add `deps/libtensorflow/lib` to `PATH`, or copy `tensorflow.dll` next to the built `.node`.

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
```

`check`/`lint` (biome) cover all JS/TS/JSON under `src/`, `scripts/`, and the root configs; `check:cpp` covers the C++ addon with a pinned `clang-format` (config in `.clang-format`, run via `uvx` — no separate install). `check:all` runs both. CI runs biome on every OS and the C++ check on Linux.

Tests that need the compiled addon are **skip-guarded**: they run automatically once `bun run build:native` has produced the addon, and are skipped otherwise, so `bun test` stays green without a native build.

## Adding a platform / bumping libtensorflow

Checksums are pinned in `scripts/install.mjs`. To add one:

1. Set `TFJS_NATIVE_LIBTENSORFLOW_VERSION` (and platform, if cross-fetching) and run `node scripts/install.mjs`.
2. The installer prints `sha256(<artifact>) = <hash>` for unpinned artifacts.
3. Add that entry to the `CHECKSUMS` map and commit.

Bumping the default version follows the **libtensorflow Version Policy** in [docs/RULES.md](./docs/RULES.md) (how it maps to a tfjs-native version bump).

## Regenerating op wrappers

Op wrappers under `src/ops/generated/` are produced by `scripts/codegen/` from the runtime op registry (`TF_GetAllOpList`). **Do not hand-edit generated files** — change the generator/template and run `bun run codegen`.

## Pull requests

- Branch from `main` (`feat/…`, `fix/…`, `chore/…`); never push to `main` directly.
- Commit messages follow **Conventional Commits**, in English (see [docs/RULES.md](./docs/RULES.md)). Example: `feat(ops): add scatterNd wrapper`.
- CI must be green: native build (where applicable), `typecheck`, `biome`, `bun test`.
- PRs that change the public API must include the reviewed `d.ts` impact.
- Keep generated code and hand-written changes in separate commits.

## Releasing (maintainers)

Releases run on a pushed tag `v<semver>` via `.github/workflows/release.yml`: prebuilds are built per OS, aggregated, and published to npm via **trusted publishing (OIDC)** — no `NPM_TOKEN`. Ensure all platform prebuilds are present before tagging.
