// Shared Python resolution for node-gyp, used by both `bun run setup` and
// `bun run build:native` so either one works standalone — a broken/missing
// system Python must not block a plain `bun run build:native`.

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

function capture(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, encoding: "utf8" });
  if (result.error !== undefined || result.status !== 0) return null;
  const out = result.stdout.trim();
  return out.length > 0 ? out : null;
}

function has(command, cwd) {
  return capture(command, ["--version"], cwd) !== null;
}

/**
 * Resolves a Python executable for node-gyp: respects an already-set `PYTHON`,
 * otherwise installs/finds the version pinned in `.python-version` via uv.
 * Returns the path, or `undefined` to leave discovery to node-gyp itself
 * (e.g. uv is not installed).
 */
export function resolvePython(root) {
  const fromEnv = process.env.PYTHON;
  if (fromEnv !== undefined && existsSync(fromEnv)) {
    console.log(`using PYTHON=${fromEnv}`);
    return fromEnv;
  }

  const versionFile = join(root, ".python-version");
  const version = existsSync(versionFile) ? readFileSync(versionFile, "utf8").trim() : undefined;

  if (!has("uv", root)) {
    console.log("uv not found — leaving Python discovery to node-gyp.");
    console.log("If the build fails, install uv (https://docs.astral.sh/uv/) and re-run.");
    return undefined;
  }

  // `uv python install` with no argument reads .python-version.
  const install = spawnSync("uv", ["python", "install"], { cwd: root, stdio: "inherit" });
  if (install.error !== undefined || install.status !== 0) {
    console.log("uv could not install the pinned Python — leaving discovery to node-gyp.");
    return undefined;
  }

  const found = version !== undefined ? capture("uv", ["python", "find", version], root) : null;
  if (found === null) {
    console.log("uv could not report a Python path — leaving discovery to node-gyp.");
    return undefined;
  }
  console.log(`using ${found}`);
  return found;
}
