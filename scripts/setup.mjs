// One-command developer setup: installs JS deps, resolves a Python for node-gyp,
// fetches libtensorflow, builds the native addon, and verifies it loads.
//
//   bun run setup
//
// Everything here is plain Node with no imports beyond the standard library, so
// it also works before `bun install` has ever run.

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

// Every tool we spawn (node, bun, uv) is a real executable, so we never need a
// shell — which also avoids Node's shell-argument-escaping deprecation.
let step = 0;
const heading = (text) => console.log(`\n[${++step}] ${text}`);

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
    ...options,
    env: { ...process.env, ...options.env },
  });
  if (result.error !== undefined || result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed`);
  }
}

/**
 * Runs JS in a real Node process, passing the source on stdin so no shell can
 * mangle it. `node` is a real executable, so it needs no shell to resolve.
 */
function runNodeScript(source, env = {}) {
  const result = spawnSync("node", ["-"], {
    cwd: root,
    input: source,
    stdio: ["pipe", "inherit", "inherit"],
    env: { ...process.env, ...env },
  });
  if (result.error !== undefined || result.status !== 0) {
    throw new Error("addon smoke test failed");
  }
}

function capture(command, args) {
  const result = spawnSync(command, args, { cwd: root, encoding: "utf8" });
  if (result.error !== undefined || result.status !== 0) return null;
  const out = result.stdout.trim();
  return out.length > 0 ? out : null;
}

function has(command) {
  return capture(command, ["--version"]) !== null;
}

/** Finds a Python for node-gyp, installing the pinned version through uv if possible. */
function resolvePython() {
  const fromEnv = process.env.PYTHON;
  if (fromEnv !== undefined && existsSync(fromEnv)) {
    console.log(`using PYTHON=${fromEnv}`);
    return fromEnv;
  }

  const versionFile = join(root, ".python-version");
  const version = existsSync(versionFile) ? readFileSync(versionFile, "utf8").trim() : undefined;

  if (!has("uv")) {
    console.log("uv not found — leaving Python discovery to node-gyp.");
    console.log("If the build fails, install uv (https://docs.astral.sh/uv/) and re-run.");
    return undefined;
  }

  // `uv python install` with no argument reads .python-version.
  run("uv", ["python", "install"]);
  const found = version !== undefined ? capture("uv", ["python", "find", version]) : null;
  if (found === null) {
    console.log("uv could not report a Python path — leaving discovery to node-gyp.");
    return undefined;
  }
  console.log(`using ${found}`);
  return found;
}

function main() {
  console.log("tfjs-native setup");

  heading("Install JS dependencies");
  if (existsSync(join(root, "node_modules"))) {
    console.log("node_modules present — skipping.");
  } else if (has("bun")) {
    // libtensorflow is fetched explicitly below, so skip it during install.
    run("bun", ["install"], { env: { TFJS_NATIVE_SKIP_INSTALL: "1" } });
  } else {
    throw new Error("bun is required (https://bun.sh)");
  }

  heading("Resolve Python for node-gyp");
  const python = resolvePython();

  heading("Fetch libtensorflow");
  run("node", [join("scripts", "install.mjs")]);

  heading("Build the native addon");
  const gyp = join("node_modules", "node-gyp", "bin", "node-gyp.js");
  if (!existsSync(join(root, gyp))) {
    throw new Error("node-gyp is missing — run `bun install` first");
  }
  // Spawn a real Node: this script may be running under Bun, and node-gyp
  // targets the headers of whichever runtime launches it.
  run("node", [gyp, "rebuild"], {
    env: python !== undefined ? { PYTHON: python, npm_config_python: python } : {},
  });

  heading("Verify the addon loads");
  // Mirrors src/backend/native.ts: Windows has no rpath, so tensorflow.dll is
  // found through PATH at load time.
  runNodeScript(`
    const path = require("node:path");
    const root = process.cwd();
    if (process.platform === "win32") {
      const lib = path.join(root, "deps", "libtensorflow", "lib");
      process.env.PATH = lib + ";" + (process.env.PATH || "");
    }
    const binding = require("node-gyp-build")(root);
    console.log("libtensorflow " + binding.version());
  `);

  console.log("\nSetup complete. Try: bun test");
}

try {
  main();
} catch (error) {
  console.error(`\nsetup failed: ${error instanceof Error ? error.message : error}`);
  process.exit(1);
}
