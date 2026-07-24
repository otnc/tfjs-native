# tfjs-native

**English** | [日本語](./README-ja.md)

[![CI](https://github.com/otnc/tfjs-native/actions/workflows/ci.yml/badge.svg)](https://github.com/otnc/tfjs-native/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/tfjs-native.svg)](https://www.npmjs.com/package/tfjs-native)
![node](https://img.shields.io/badge/node-%3E%3D22-brightgreen)
![platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

TensorFlow (C++/Python) native wrapper for Node.js.

`@tensorflow/tfjs` maps only ~300 of TensorFlow's ops; the original op registry has 1,400+. **tfjs-native** binds TensorFlow's **C API** (the native library `@tensorflow/tfjs-node` also uses) to expose the **full op registry** to typed TypeScript, plus SavedModel execution and a thin training layer.

Original: [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)

## Requirements

- **Node.js 22+**, on **Windows / Linux / macOS**.
- Prebuilt addons ship for **linux-x64**, **darwin-x64**, and **win32-x64**; other platforms build from source (needs Python and a C++ toolchain).
- `libtensorflow` (the C library) is fetched automatically on install — no manual setup on supported platforms.

## Install

```sh
npm install tfjs-native
# or: bun add tfjs-native
```

The matching `libtensorflow` build downloads on install.

- **Allow the install script.** Some package managers skip a dependency's install script by default, which skips the download. Allow it for tfjs-native, or set `LIBTENSORFLOW_ROOT` to an existing install:
  - **Bun**: add `"tfjs-native"` to `trustedDependencies`.
  - **pnpm**: add `"tfjs-native"` to `pnpm.onlyBuiltDependencies`.
  - **npm**: runs it today, but is expected to require opt-in in a future release.
- **macOS (Apple Silicon)**: no official arm64 build — `brew install libtensorflow`, then set `LIBTENSORFLOW_ROOT`.
- Use a mirror with `TFJS_NATIVE_CDN_STORAGE=https://your-mirror/`.

## Usage

```ts
import { tensor, zeros, tidy, version } from "tfjs-native";

console.log(version()); // linked libtensorflow version

const a = tensor([
  [1, 2, 3],
  [4, 5, 6],
]);
console.log(a.shape); // [2, 3]
console.log(await a.array()); // [[1, 2, 3], [4, 5, 6]]

// int64 is first-class (read back as BigInt), unlike tfjs, which stops at int32.
const ids = tensor([1, 2, 3], [3], "int64");

// Scope-based memory management.
const result = tidy(() => {
  const z = zeros([2, 2]); // disposed at scope exit
  return tensor([1, 1]); // kept
});

a.dispose();
```

## How it works

```
TypeScript API  ->  N-API addon (node-addon-api)  ->  libtensorflow C API
```

The addon links a pinned libtensorflow (currently 2.10.0), is compiled per platform, and ships as a prebuilt binary via `node-gyp-build`.

## Documentation

- [Status](./docs/STATUS.md) — what works today, by milestone.
- [Functions & features](./docs/FUNCTIONS.md) — the current API and the roadmap.
- [Design](./docs/DESIGN.md) — architecture, scope, and the native contract.
- [Contributing](./CONTRIBUTING.md) — dev environment (Bun, Python via uv, C++ toolchain) and workflow.

## Author

otoneko. https://github.com/otnc

## License

[Apache 2.0](./LICENSE).
