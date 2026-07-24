// Builds a prebuilt binary for the current OS/arch (used locally and by
// release.yml). Calls prebuildify's JS API directly instead of its CLI, which
// sidesteps two Windows-specific gaps:
//
//   1. prebuildify's CLI spawns node-gyp by shelling out to the literal name
//      `node-gyp.cmd` (see its npmbin() helper), relying on PATH resolution.
//      bun's node_modules/.bin shims on Windows are .exe/.bunx, not .cmd, so
//      that spawn fails with "'node-gyp.cmd' is not recognized" even though
//      node-gyp itself is installed and working. Passing `nodeGyp` as an
//      explicit path in the options bypasses that lookup entirely.
//   2. Resolves Python for that node-gyp via uv (scripts/lib/python.mjs), the
//      same as build-native.mjs, so this works standalone without a system
//      Python (see docs/RULES.md libtensorflow/build tooling policy).
//
//   node scripts/prebuildify.mjs   # napi + strip, matching the old CLI call

import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import prebuildify from "prebuildify";
import { resolvePython } from "./lib/python.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

function main() {
  const python = resolvePython(root);
  process.env.PYTHON = python;
  process.env.npm_config_python = python;

  const opts = { cwd: root, napi: true, strip: true, targets: [] };

  if (process.platform === "win32") {
    const nodeGypExe = join(root, "node_modules", ".bin", "node-gyp.exe");
    if (existsSync(nodeGypExe)) {
      opts.nodeGyp = nodeGypExe;
    }
  }

  prebuildify(opts, (error) => {
    if (error) {
      console.error(`prebuildify failed: ${error.message ?? error}`);
      process.exit(1);
    }
  });
}

main();
