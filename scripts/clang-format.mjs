// Runs the pinned clang-format over the C++ sources via uv, so contributors need
// no separate clang-format install. `uvx` may not be on PATH even when uv is, so
// we resolve a uv binary ourselves and call `uv tool run`.
//
//   node scripts/clang-format.mjs --check   # dry-run, non-zero if unformatted
//   node scripts/clang-format.mjs --write    # format in place
//
// See docs/RULES.md (formatting) and .clang-format for the style.

import { spawnSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { delimiter, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const CLANG_FORMAT = "clang-format@22.1.8";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const nativeDir = join(root, "src", "native");

/** Locates a `uv` executable: PATH first, then the common install locations. */
function findUv() {
  const exe = process.platform === "win32" ? "uv.exe" : "uv";
  const probe = (dir) => {
    const candidate = join(dir, exe);
    return existsSync(candidate) ? candidate : null;
  };

  for (const dir of (process.env.PATH ?? "").split(delimiter)) {
    if (dir && probe(dir)) return exe; // on PATH; let the OS resolve it
  }
  const home = process.env.USERPROFILE ?? process.env.HOME ?? "";
  const candidates = [
    join(home, ".local", "bin"),
    join(home, ".cargo", "bin"),
    "D:\\devtools\\uv\\bin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
  ];
  for (const dir of candidates) {
    const found = probe(dir);
    if (found) return found;
  }
  return null;
}

function sources() {
  return readdirSync(nativeDir)
    .filter((f) => f.endsWith(".cc") || f.endsWith(".h"))
    .map((f) => join(nativeDir, f));
}

function main() {
  const write = process.argv.includes("--write");
  const uv = findUv();
  if (uv === null) {
    console.error("clang-format: uv not found. Install uv (https://docs.astral.sh/uv/).");
    process.exit(1);
  }

  const files = sources();
  const args = ["tool", "run", CLANG_FORMAT, write ? "-i" : "--dry-run", "--Werror", ...files];
  const result = spawnSync(uv, args, { cwd: root, stdio: "inherit" });
  if (result.error !== undefined) {
    console.error(`clang-format: ${result.error.message}`);
    process.exit(1);
  }
  process.exit(result.status ?? 1);
}

main();
