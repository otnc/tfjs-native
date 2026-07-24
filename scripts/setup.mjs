// One-command developer setup: installs JS deps, fetches libtensorflow, builds
// the native addon (via build-native.mjs, which resolves Python for node-gyp),
// and verifies it loads.
//
//   bun run setup
//
// Everything here is plain Node with no imports beyond the standard library, so
// it also works before `bun install` has ever run.

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
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

  heading("Fetch libtensorflow");
  run("node", [join("scripts", "install.mjs")]);

  heading("Build the native addon");
  // Delegates to build-native.mjs, which resolves a Python for node-gyp via uv
  // (see scripts/lib/python.mjs) so this works even with no usable system Python.
  run("node", [join("scripts", "build-native.mjs")]);

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
