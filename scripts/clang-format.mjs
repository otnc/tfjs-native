// Runs the pinned clang-format over the C++ sources via uv, so contributors need
// no separate clang-format install. Assumes `uvx` is on PATH (a prerequisite —
// see CONTRIBUTING.md); the official uv installer puts it there by default.
//
//   node scripts/clang-format.mjs --check   # dry-run, non-zero if unformatted
//   node scripts/clang-format.mjs --write   # format in place
//
// See docs/RULES.md (formatting) and .clang-format for the style.

import { spawnSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const CLANG_FORMAT = "clang-format@22.1.8";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const nativeDir = join(root, "src", "native");

function sources() {
  return readdirSync(nativeDir)
    .filter((f) => f.endsWith(".cc") || f.endsWith(".h"))
    .map((f) => join(nativeDir, f));
}

function main() {
  const write = process.argv.includes("--write");
  const files = sources();
  const args = [CLANG_FORMAT, write ? "-i" : "--dry-run", "--Werror", ...files];
  const result = spawnSync("uvx", args, { cwd: root, stdio: "inherit" });
  if (result.error !== undefined) {
    console.error(`clang-format: ${result.error.message}`);
    console.error("Is uv installed and on PATH? See CONTRIBUTING.md.");
    process.exit(1);
  }
  process.exit(result.status ?? 1);
}

main();
