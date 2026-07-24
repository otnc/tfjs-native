// Shared Python resolution for node-gyp, used by both `bun run setup` and
// `bun run build:native`. uv is a required, on-PATH prerequisite (see
// CONTRIBUTING.md), so this fails fast with a clear message when it is
// missing — never falls through to node-gyp's own Python discovery, which
// turns any missing/broken system Python into a long, confusing cascade of
// "could not be run" errors.

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

function run(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.error !== undefined || result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed`);
  }
}

function capture(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, encoding: "utf8" });
  if (result.error !== undefined || result.status !== 0) return null;
  const out = result.stdout.trim();
  return out.length > 0 ? out : null;
}

/**
 * Resolves a Python executable for node-gyp: an already-set `PYTHON` wins,
 * otherwise the version pinned in `.python-version` is installed/found via uv.
 * Throws if uv is missing or fails — uv is a required prerequisite here, not
 * an optional convenience.
 */
export function resolvePython(root) {
  const fromEnv = process.env.PYTHON;
  if (fromEnv !== undefined && existsSync(fromEnv)) {
    console.log(`using PYTHON=${fromEnv}`);
    return fromEnv;
  }

  if (capture("uv", ["--version"], root) === null) {
    throw new Error(
      "uv is required to resolve Python for node-gyp but was not found on PATH. " +
        "Install it (https://docs.astral.sh/uv/) and ensure it's on PATH, or set " +
        "PYTHON yourself. See CONTRIBUTING.md.",
    );
  }

  // `uv python install` with no argument reads .python-version.
  run("uv", ["python", "install"], root);

  const versionFile = join(root, ".python-version");
  const version = existsSync(versionFile) ? readFileSync(versionFile, "utf8").trim() : undefined;
  const found = version !== undefined ? capture("uv", ["python", "find", version], root) : null;
  if (found === null) {
    throw new Error(
      `uv could not find the Python it just installed (.python-version: ${version ?? "unset"}).`,
    );
  }
  console.log(`using ${found}`);
  return found;
}
