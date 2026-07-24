// Builds the native addon. Resolves a Python for node-gyp automatically via uv
// (see scripts/lib/python.mjs) when PYTHON is not already set, so plain
// `bun run build:native` works on its own — not only as part of
// `bun run setup`. Fails fast with a clear message if uv is missing, instead
// of letting node-gyp's own Python probing produce a wall of errors.
//
//   node scripts/build-native.mjs [node-gyp args...]   # default: rebuild

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { resolvePython } from "./lib/python.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const gyp = join(root, "node_modules", "node-gyp", "bin", "node-gyp.js");

function main() {
  if (!existsSync(gyp)) {
    throw new Error("node-gyp is missing — run `bun install` first");
  }

  const python = resolvePython(root);
  const args = process.argv.slice(2);
  const gypArgs = args.length > 0 ? args : ["rebuild"];

  // Spawn a real Node: this script may run under Bun, and node-gyp targets the
  // headers of whichever runtime launches it. No shell, so nothing can mangle
  // paths/arguments (see scripts/clang-format.mjs for the same concern).
  const result = spawnSync("node", [gyp, ...gypArgs], {
    cwd: root,
    stdio: "inherit",
    env: { ...process.env, PYTHON: python, npm_config_python: python },
  });
  if (result.error !== undefined) {
    throw result.error;
  }
  process.exit(result.status ?? 1);
}

try {
  main();
} catch (error) {
  console.error(`build:native failed: ${error instanceof Error ? error.message : error}`);
  process.exit(1);
}
