// Integration tests that exercise the native addon. They are skipped unless the
// N-API addon has been built (needs libtensorflow + a C++ toolchain). Once
// `bun run build:native` succeeds, these run as real round-trip tests.

import { describe, expect, it } from "bun:test";

// Probe whether the addon loads. native.ts throws at import when no prebuild /
// local build is found, so a failed dynamic import means "not built here".
let nativeAvailable = false;
try {
  await import("../backend/native.js");
  nativeAvailable = true;
} catch {
  nativeAvailable = false;
}

describe.skipIf(!nativeAvailable)("tensor round-trip (native)", () => {
  it("creates a float32 tensor and reads it back", async () => {
    const { tensor } = await import("./factory.js");
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(t.shape).toEqual([2, 3]);
    expect(t.dtype).toBe("float32");
    expect(t.size).toBe(6);
    expect(await t.array()).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    t.dispose();
    expect(t.isDisposed).toBe(true);
  });

  it("round-trips int64 through BigInt", async () => {
    const { tensor } = await import("./factory.js");
    const t = tensor([1, 2, 3], [3], "int64");
    expect(Array.from(t.dataSync() as BigInt64Array)).toEqual([1n, 2n, 3n]);
    t.dispose();
  });

  it("zeros/ones fill correctly", async () => {
    const { ones, zeros } = await import("./factory.js");
    const z = zeros([2, 2]);
    const o = ones([2, 2]);
    expect(await z.array()).toEqual([
      [0, 0],
      [0, 0],
    ]);
    expect(await o.array()).toEqual([
      [1, 1],
      [1, 1],
    ]);
    z.dispose();
    o.dispose();
  });

  it("tidy disposes intermediates but keeps the result", async () => {
    const { tensor } = await import("./factory.js");
    const { tidy } = await import("./engine.js");
    let intermediate: import("./tensor.js").Tensor | undefined;
    const kept = tidy(() => {
      intermediate = tensor([1, 2, 3]);
      return tensor([4, 5, 6]);
    });
    expect(intermediate?.isDisposed).toBe(true);
    expect(kept.isDisposed).toBe(false);
    kept.dispose();
  });
});
