import { defineConfig } from "tsdown";

export default defineConfig({
  entry: ["src/index.ts"],
  // Ship both ESM (.mjs) and CJS (.cjs) so the package works from either
  // module system; d.ts is emitted for both (.d.mts / .d.cts).
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  target: "node22",
  // The native addon is resolved at runtime via node-gyp-build; never bundle it.
  // (Dependencies are external by default; this makes the intent explicit.)
  deps: { neverBundle: ["node-gyp-build"] },
});
